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
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.util.Progress;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
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
    private String artifactId;
    private String modelName;

    private Metadata metadata;
    private boolean resolved;

    S3Repository(
            S3Client client,
            String name,
            String bucket,
            String prefix,
            String artifactId,
            String modelName) {
        this.client = client;
        this.name = name;
        this.bucket = bucket;
        this.prefix = prefix;
        this.artifactId = artifactId;
        this.modelName = modelName;
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
    public Artifact resolve(MRL mrl, Map<String, String> filter) throws IOException {
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
        logger.debug("Downloading artifact from: s3://{}/{} ...", bucket, key);
        GetObjectRequest req = GetObjectRequest.builder().bucket(bucket).key(key).build();
        try (ResponseInputStream<GetObjectResponse> is = client.getObject(req)) {
            save(is, tmp, baseUri, item, progress);
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<MRL> getResources() {
        try {
            Metadata m = getMetadata();
            if (m != null && !m.getArtifacts().isEmpty()) {
                MRL mrl = model(Application.UNDEFINED, m.getGroupId(), m.getArtifactId());
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
            Artifact artifact = listFiles();
            if (artifact == null) {
                logger.debug("No object found in s3 bucket.");
                return null;
            }

            metadata = new Metadata.MatchAllMetadata();
            String hash = md5hash("s3://" + bucket + '/' + prefix);
            MRL mrl = model(Application.UNDEFINED, DefaultModelZoo.GROUP_ID, hash);
            metadata.setRepositoryUri(mrl.toURI());
            metadata.setArtifactId(artifactId);
            metadata.setArtifacts(Collections.singletonList(artifact));
            return metadata;
        } catch (SdkException e) {
            throw new IOException("Failed scan s3 bucket: " + bucket, e);
        }
    }

    private Artifact listFiles() {
        ListObjectsRequest req =
                ListObjectsRequest.builder()
                        .bucket(bucket)
                        .maxKeys(100)
                        .prefix(prefix)
                        .delimiter("/")
                        .build();

        ListObjectsResponse resp = client.listObjects(req);
        List<S3Object> list = resp.contents();
        if (list.isEmpty()) {
            return null;
        }

        Artifact artifact = new Artifact();
        artifact.setName(modelName);
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
}
