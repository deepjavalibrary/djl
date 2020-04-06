/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.aws.s3;

import ai.djl.Application;
import ai.djl.repository.AbstractRepository;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Metadata;
import ai.djl.repository.Repository;
import ai.djl.util.Progress;
import ai.djl.util.ZipUtils;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.core.exception.SdkException;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.ListObjectsRequest;
import software.amazon.awssdk.services.s3.model.ListObjectsResponse;
import software.amazon.awssdk.services.s3.model.S3Object;

/**
 * A {@code S3Repository} is a {@link Repository} located on a AWS S3.
 *
 * @see Repository
 */
public class S3Repository extends AbstractRepository {

    private static final Logger logger = LoggerFactory.getLogger(S3Repository.class);

    private S3Client client;
    private String name;
    private String bucket;
    private String prefix;

    private Metadata metadata;
    private boolean resolved;

    S3Repository(S3Client client, String name, String bucket, String prefix) {
        this.client = client;
        this.name = name;
        this.bucket = bucket;
        this.prefix = prefix;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isRemote() {
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return name;
    }

    /** {@inheritDoc} */
    @Override
    public URI getBaseUri() {
        URI uri = URI.create(bucket);
        if (prefix.isEmpty()) {
            return uri;
        }
        return uri.resolve(prefix);
    }

    /** {@inheritDoc} */
    @Override
    public Metadata locate(MRL mrl) throws IOException {
        return getMetadata();
    }

    /** {@inheritDoc} */
    @Override
    public Artifact resolve(MRL mrl, String version, Map<String, String> filter)
            throws IOException {
        Metadata m = locate(mrl);
        if (m == null) {
            return null;
        }
        List<Artifact> artifacts = m.getArtifacts();
        if (artifacts.isEmpty()) {
            return null;
        }
        return artifacts.get(0);
    }

    /** {@inheritDoc} */
    @Override
    protected void download(Path tmp, URI baseUri, Artifact.Item item, Progress progress)
            throws IOException {
        String key = item.getUri();
        GetObjectRequest req = GetObjectRequest.builder().bucket(bucket).key(key).build();
        try (ResponseInputStream<GetObjectResponse> is = client.getObject(req)) {
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
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<MRL> getResources() {
        try {
            Metadata m = getMetadata();
            if (m != null && !m.getArtifacts().isEmpty()) {
                MRL mrl = MRL.model(Application.UNDEFINED, m.getGroupId(), m.getArtifactId());
                return Collections.singletonList(mrl);
            }
        } catch (IOException e) {
            logger.warn("Failed to scan S3: " + bucket, e);
        }
        return Collections.emptyList();
    }

    private synchronized Metadata getMetadata() throws IOException {
        if (resolved) {
            return metadata;
        }

        try {
            resolved = true;
            String artifactId;
            String modelName;
            String key;
            if (prefix.isEmpty()) {
                key = "/";
                artifactId = bucket;
                modelName = artifactId;
            } else if (prefix.endsWith("/")) {
                key = prefix;
                String[] tokens = prefix.split("/");
                artifactId = tokens[tokens.length - 1];
                modelName = artifactId;
            } else {
                List<S3Object> list = listFiles(prefix, 1, false);
                if (list.isEmpty()) {
                    logger.debug("No object found in s3 bucket: " + prefix);
                    return null;
                }
                S3Object obj = list.get(0);
                String objName = obj.key();
                if (objName.endsWith("/")) {
                    objName = objName.substring(1, objName.length() - 1);
                }
                int pos = objName.indexOf('/', prefix.length() + 1);
                if (pos == -1) {
                    pos = objName.lastIndexOf('/');
                }

                key = objName.substring(0, pos + 1);
                if ("/".equals(key)) {
                    artifactId = bucket;
                    modelName = artifactId;
                } else {
                    String[] tokens = key.split("/");
                    artifactId = tokens[tokens.length - 1];
                    if (key.length() < prefix.length()) {
                        modelName = prefix.substring(key.length());
                    } else {
                        modelName = artifactId;
                    }
                }
            }

            List<S3Object> list = listFiles(key, 100, true); // only list one level
            if (list.isEmpty()) {
                logger.debug("No object found in s3 bucket.");
                return null;
            }

            metadata = new Metadata.MatchAllMetadata();
            MRL mrl = MRL.model(Application.UNDEFINED, metadata.getGroupId(), artifactId);
            metadata.setRepositoryUri(mrl.toURI());
            metadata.setArtifactId(artifactId);
            Artifact artifact = createArtifact(list);
            artifact.setName(modelName);
            metadata.setArtifacts(Collections.singletonList(artifact));
            return metadata;
        } catch (SdkException e) {
            throw new IOException("Failed scan s3 bucket: " + bucket, e);
        }
    }

    private List<S3Object> listFiles(String key, int maxKeys, boolean delimiter) {
        ListObjectsRequest.Builder builder =
                ListObjectsRequest.builder().bucket(bucket).maxKeys(maxKeys).prefix(key);
        if (delimiter) {
            builder.delimiter("/");
        }

        ListObjectsRequest req = builder.build();

        ListObjectsResponse resp = client.listObjects(req);
        return resp.contents();
    }

    private Artifact createArtifact(List<S3Object> list) {
        Artifact artifact = new Artifact();
        Map<String, Artifact.Item> files = new ConcurrentHashMap<>();
        for (S3Object obj : list) {
            Artifact.Item item = new Artifact.Item();
            String key = obj.key();
            if (!key.endsWith("/")) {
                item.setUri(key);
                item.setSize(obj.size());
                item.setArtifact(artifact);
                if ("dir".equals(item.getType())) {
                    item.setName(""); // avoid creating extra folder
                }
                files.put(key, item);
            }
        }
        artifact.setFiles(files);
        return artifact;
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
}
