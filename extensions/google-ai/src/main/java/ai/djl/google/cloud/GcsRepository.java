/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.google.cloud;

import ai.djl.Application;
import ai.djl.repository.AbstractRepository;
import ai.djl.repository.Artifact;
import ai.djl.repository.FilenameUtils;
import ai.djl.repository.MRL;
import ai.djl.repository.Metadata;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.util.Progress;
import ai.djl.util.Utils;

import com.google.api.gax.paging.Page;
import com.google.cloud.ReadChannel;
import com.google.cloud.storage.Blob;
import com.google.cloud.storage.BlobId;
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.channels.Channels;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@code GcsRepository} is a {@link Repository} located on Google Cloud Storage.
 *
 * @see Repository
 */
public class GcsRepository extends AbstractRepository {

    private static final Logger logger = LoggerFactory.getLogger(GcsRepository.class);

    private Storage storage;
    private String bucket;
    private String prefix;
    private String artifactId;
    private String modelName;

    private Metadata metadata;
    private boolean resolved;

    GcsRepository(String name, URI uri, Storage storage) {
        super(name, uri);
        this.storage = storage;

        bucket = uri.getHost();
        prefix = uri.getPath();
        if (!prefix.isEmpty()) {
            prefix = prefix.substring(1);
        }
        boolean isArchive = FilenameUtils.isArchiveFile(prefix);
        if (!isArchive && !prefix.isEmpty() && !prefix.endsWith("/")) {
            prefix += '/'; // NOPMD
        }

        modelName = arguments.get("model_name");
        artifactId = arguments.get("artifact_id");
        if (artifactId == null) {
            if (prefix.isEmpty()) {
                artifactId = bucket;
            } else {
                Path path = Paths.get(prefix);
                Path fileName = path.getFileName();
                if (fileName == null) {
                    throw new AssertionError("This should never happen.");
                }
                artifactId = fileName.toString();
                if (isArchive) {
                    artifactId = FilenameUtils.getNamePart(artifactId);
                }
            }
        }

        if (modelName == null) {
            modelName = artifactId;
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean isRemote() {
        return true;
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
        logger.debug("Downloading artifact from: gs://{}/{} ...", bucket, key);
        Blob blob = storage.get(BlobId.of(bucket, key));
        if (blob == null) {
            throw new IOException("Object not found: gs://" + bucket + '/' + key);
        }
        try (ReadChannel reader = blob.reader();
                InputStream is = Channels.newInputStream(reader)) {
            save(is, tmp, item, progress);
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
            logger.warn("Failed to scan GCS bucket: {}", bucket, e);
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
                logger.debug("No object found in gcs bucket.");
                return null;
            }

            metadata = new Metadata.MatchAllMetadata();
            String hash = Utils.hash("gs://" + bucket + '/' + prefix);
            MRL mrl = model(Application.UNDEFINED, DefaultModelZoo.GROUP_ID, hash);
            metadata.setRepositoryUri(mrl.toURI());
            metadata.setArtifactId(artifactId);
            metadata.setArtifacts(Collections.singletonList(artifact));
            return metadata;
        } catch (StorageException e) {
            throw new IOException("Failed scan gcs bucket: " + bucket, e);
        }
    }

    private Artifact listFiles() {
        Page<Blob> blobs =
                storage.list(
                        bucket,
                        Storage.BlobListOption.prefix(prefix),
                        Storage.BlobListOption.currentDirectory(),
                        Storage.BlobListOption.pageSize(100));

        Artifact artifact = new Artifact();
        artifact.setName(modelName);
        artifact.getArguments().putAll(arguments);
        Map<String, Artifact.Item> files = new ConcurrentHashMap<>();
        for (Blob blob : blobs.iterateAll()) {
            Artifact.Item item = new Artifact.Item();
            String key = blob.getName();
            if (!key.endsWith("/")) {
                item.setUri(key);
                item.setSize(blob.getSize() == null ? 0 : blob.getSize());
                item.setArtifact(artifact);
                if ("dir".equals(item.getType())) {
                    item.setName(""); // avoid creating extra folder
                }
                files.put(key, item);
            }
        }
        if (files.isEmpty()) {
            return null;
        }
        artifact.setFiles(files);
        return artifact;
    }
}
