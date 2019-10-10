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

import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipInputStream;

public abstract class AbstractRepository implements Repository {

    @Override
    public InputStream openStream(Artifact.Item item, String path) throws IOException {
        return Files.newInputStream(Paths.get(resolvePath(item, path)));
    }

    @Override
    public String[] listDirectory(Artifact.Item item, String path) throws IOException {
        return Paths.get(resolvePath(item, path)).toFile().list();
    }

    @Override
    public Path getFile(Artifact.Item item, String path) throws IOException {
        return Paths.get(resolvePath(item, path)).toAbsolutePath();
    }

    private URI resolvePath(Artifact.Item item, String path) throws IOException {
        Artifact artifact = item.getArtifact();
        URI artifactUri = artifact.getResourceUri();

        String itemUri = item.getUri();
        // Resolve cached item
        if (itemUri != null && URI.create(itemUri).isAbsolute()) {
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

    @Override
    public void prepare(Artifact artifact) throws IOException {
        Path cacheDir = getCacheDirectory();
        URI resourceUri = artifact.getResourceUri();
        Path resourceDir = cacheDir.resolve(resourceUri.getPath());
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
        try {
            for (Artifact.Item item : files.values()) {
                download(tmp, baseUri, item);
            }
            Files.move(tmp, resourceDir, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Utils.deleteQuietly(tmp);
        }
    }

    @Override
    public Path getCacheDirectory() throws IOException {
        String cacheDir = System.getProperty("JOULE_CACHE_DIR");
        if (cacheDir == null) {
            cacheDir = System.getenv("JOULE_CACHE_DIR");
            if (cacheDir == null) {
                String userHome = System.getProperty("user.home");
                cacheDir = userHome + "/.joule/cache";
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

    private void download(Path tmp, URI baseUri, Artifact.Item item) throws IOException {
        URI fileUri = URI.create(item.getUri());
        if (!fileUri.isAbsolute()) {
            fileUri = getBaseUri().resolve(baseUri).resolve(fileUri);
        }

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
            if (!"zip".equals(extension)) {
                throw new IOException("File type is not supported: " + extension);
            }
            try (InputStream is = fileUri.toURL().openStream()) {
                ZipUtils.unzip(is, dir);
            }
            return;
        }

        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA1");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError("SHA1 algorithm not found.", e);
        }

        try (InputStream is = fileUri.toURL().openStream();
                DigestInputStream dis = new DigestInputStream(is, md)) {
            Path file = tmp.resolve(fileName);
            if ("zip".equals(extension)) {
                try (ZipInputStream zis = new ZipInputStream(dis)) {
                    zis.getNextEntry();
                    Files.copy(zis, file);
                }
            } else if ("gzip".equals(extension)) {
                try (GZIPInputStream zis = new GZIPInputStream(dis)) {
                    Files.copy(zis, file);
                }
            } else if (extension.isEmpty()) {
                Files.copy(dis, file);
            } else {
                throw new IOException("File type is not supported: " + extension);
            }
        }

        String sha1 = Hex.toHexString(md.digest());
        if (!sha1.equalsIgnoreCase(item.getSha1Hash())) {
            throw new IOException("Checksum error: " + item.getName() + ", sha1: " + sha1);
        }
    }
}
