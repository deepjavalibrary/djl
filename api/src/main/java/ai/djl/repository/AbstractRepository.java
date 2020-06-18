/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.repository;

import ai.djl.util.Hex;
import ai.djl.util.Progress;
import ai.djl.util.Utils;
import ai.djl.util.ZipUtils;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code AbstractRepository} is the shared base for implementers of the {@link Repository}
 * interface.
 *
 * @see Repository
 */
public abstract class AbstractRepository implements Repository {

    private static final Logger logger = LoggerFactory.getLogger(AbstractRepository.class);

    /** {@inheritDoc} */
    @Override
    public InputStream openStream(Artifact.Item item, String path) throws IOException {
        return Files.newInputStream(Paths.get(resolvePath(item, path)));
    }

    /** {@inheritDoc} */
    @Override
    public String[] listDirectory(Artifact.Item item, String path) throws IOException {
        return Paths.get(resolvePath(item, path)).toFile().list();
    }

    /** {@inheritDoc} */
    @Override
    public Path getFile(Artifact.Item item, String path) throws IOException {
        return Paths.get(resolvePath(item, path)).toAbsolutePath();
    }

    protected URI resolvePath(Artifact.Item item, String path) throws IOException {
        Artifact artifact = item.getArtifact();
        URI artifactUri = artifact.getResourceUri();

        String itemUri = item.getUri();
        // Resolve cached item
        if (itemUri != null && URI.create(itemUri).isAbsolute() || isRemote()) {
            Path cacheDir = getCacheDirectory();
            Path resourceDir = cacheDir.resolve(artifactUri.getPath());
            String type = item.getType();
            String fileName = item.getName();
            Path cachedFile;
            if ("dir".equals(type)) {
                if (!fileName.isEmpty()) {
                    cachedFile = resourceDir.resolve(fileName);
                } else {
                    cachedFile = resourceDir;
                }
                return cachedFile.resolve(path).toUri();
            } else {
                return resourceDir.resolve(fileName).toUri();
            }
        }

        // Resolve metadata item
        String uriSuffix = itemUri != null ? itemUri : item.getName();
        return getBaseUri().resolve(artifactUri.resolve(uriSuffix));
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Artifact artifact, Progress progress) throws IOException {
        Path resourceDir = getResourceDirectory(artifact);
        if (Files.exists(resourceDir)) {
            // files have been downloaded already.
            return;
        }

        Metadata metadata = artifact.getMetadata();
        URI baseUri = metadata.getRepositoryUri();
        Map<String, Artifact.Item> files = artifact.getFiles();

        Path parentDir = resourceDir.toAbsolutePath().getParent();
        if (parentDir == null) {
            throw new AssertionError("Parent path should never be null: " + resourceDir.toString());
        }

        Files.createDirectories(parentDir);
        Path tmp = Files.createTempDirectory(parentDir, resourceDir.toFile().getName());
        if (progress != null) {
            long totalSize = 0;
            for (Artifact.Item item : files.values()) {
                totalSize += item.getSize();
            }
            progress.reset("Downloading", totalSize);
        }

        try {
            for (Artifact.Item item : files.values()) {
                download(tmp, baseUri, item, progress);
            }
            Files.move(tmp, resourceDir, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Utils.deleteQuietly(tmp);
            if (progress != null) {
                progress.end();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public Path getCacheDirectory() throws IOException {
        String cacheDir = System.getProperty("DJL_CACHE_DIR");
        if (cacheDir == null || cacheDir.isEmpty()) {
            cacheDir = System.getenv("DJL_CACHE_DIR");
            if (cacheDir == null || cacheDir.isEmpty()) {
                String userHome = System.getProperty("user.home");
                cacheDir = userHome + "/.djl.ai/cache";
            }
        }

        Path dir = Paths.get(cacheDir, "repo");
        if (Files.notExists(dir)) {
            Files.createDirectories(dir);
        } else if (!Files.isDirectory(dir)) {
            throw new IOException("Failed initialize cache directory: " + dir.toString());
        }
        return dir;
    }

    /** {@inheritDoc} */
    @Override
    public List<MRL> getResources() {
        return Collections.emptyList();
    }

    protected void download(Path tmp, URI baseUri, Artifact.Item item, Progress progress)
            throws IOException {
        URI fileUri = URI.create(item.getUri());
        if (!fileUri.isAbsolute()) {
            fileUri = getBaseUri().resolve(baseUri).resolve(fileUri);
        }

        logger.debug("Downloading artifact: {} ...", fileUri);
        try (InputStream is = fileUri.toURL().openStream()) {
            save(is, tmp, baseUri, item, progress);
        }
    }

    protected void save(
            InputStream is, Path tmp, URI baseUri, Artifact.Item item, Progress progress)
            throws IOException {
        ProgressInputStream pis = new ProgressInputStream(is, progress);
        String fileName = item.getName();
        String extension = item.getExtension();
        if ("dir".equals(item.getType())) {
            Path dir;
            if (!fileName.isEmpty()) {
                // honer the name set in metadata.json
                dir = tmp.resolve(fileName);
                Files.createDirectories(dir);
            } else {
                dir = tmp;
            }
            if ("zip".equals(extension)) {
                ZipUtils.unzip(pis, dir);
            } else if ("tgz".equals(extension)) {
                untar(pis, dir, true);
            } else if ("tar".equals(extension)) {
                untar(pis, dir, false);
            } else {
                throw new IOException("File type is not supported: " + extension);
            }
        } else {
            Path file = tmp.resolve(fileName);
            if ("zip".equals(extension)) {
                ZipInputStream zis = new ZipInputStream(pis);
                zis.getNextEntry();
                Files.copy(zis, file);
            } else if ("gzip".equals(extension)) {
                Files.copy(new GZIPInputStream(pis), file);
            } else {
                Files.copy(pis, file);
            }
        }
        pis.validateChecksum(item);
    }

    private void untar(InputStream is, Path dir, boolean gzip) throws IOException {
        InputStream bis;
        if (gzip) {
            bis = new GzipCompressorInputStream(new BufferedInputStream(is));
        } else {
            bis = new BufferedInputStream(is);
        }
        try (TarArchiveInputStream tis = new TarArchiveInputStream(bis)) {
            TarArchiveEntry entry;
            while ((entry = tis.getNextTarEntry()) != null) {
                Path file = dir.resolve(entry.getName()).toAbsolutePath();
                if (entry.isDirectory()) {
                    Files.createDirectories(file);
                } else {
                    Path parentFile = file.getParent();
                    if (parentFile == null) {
                        throw new AssertionError(
                                "Parent path should never be null: " + file.toString());
                    }
                    Files.createDirectories(parentFile);
                    Files.copy(tis, file);
                }
            }
        }
    }

    protected static String md5hash(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            return Hex.toHexString(md.digest(input.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError("MD5 algorithm not found.", e);
        }
    }

    /**
     * A {@code ProgressInputStream} is a wrapper around an {@link InputStream} that also uses
     * {@link Progress}.
     */
    private static final class ProgressInputStream extends InputStream {

        private DigestInputStream dis;
        private Progress progress;

        /**
         * Constructs a new ProgressInputStream with an input stream and progress.
         *
         * @param is the input stream
         * @param progress the (optionally null) progress tracker
         */
        public ProgressInputStream(InputStream is, Progress progress) {
            MessageDigest md;
            try {
                md = MessageDigest.getInstance("SHA1");
            } catch (NoSuchAlgorithmException e) {
                throw new AssertionError("SHA1 algorithm not found.", e);
            }
            dis = new DigestInputStream(is, md);
            this.progress = progress;
        }

        /** {@inheritDoc} */
        @Override
        public int read() throws IOException {
            int ret = dis.read();
            if (progress != null) {
                if (ret >= 0) {
                    progress.increment(1);
                } else {
                    progress.end();
                }
            }
            return ret;
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            int size = dis.read(b, off, len);
            if (progress != null) {
                progress.increment(size);
            }
            return size;
        }

        private void validateChecksum(Artifact.Item item) throws IOException {
            String expectedHash = item.getSha1Hash();
            if (expectedHash == null) {
                return;
            }

            // drain InputSteam to get correct sha1 hash
            Utils.toByteArray(dis);
            String sha1 = Hex.toHexString(dis.getMessageDigest().digest());
            if (!sha1.equalsIgnoreCase(item.getSha1Hash())) {
                throw new IOException(
                        "Checksum error: "
                                + item.getName()
                                + ", expected sha1: "
                                + item.getSha1Hash()
                                + ", actual sha1: "
                                + sha1);
            }
        }

        /** {@inheritDoc} */
        @Override
        public void close() throws IOException {
            dis.close();
        }
    }
}
