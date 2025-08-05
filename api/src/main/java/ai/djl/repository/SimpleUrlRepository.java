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
package ai.djl.repository;

import ai.djl.Application;
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.util.Progress;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@code SimpleUrlRepository} is a {@link Repository} contains an archive file from a HTTP URL.
 *
 * @see Repository
 */
public class SimpleUrlRepository extends AbstractRepository {

    private static final Logger logger = LoggerFactory.getLogger(SimpleUrlRepository.class);

    private String fileName;
    private String artifactId;
    private String modelName;

    private Metadata metadata;
    private boolean resolved;

    SimpleUrlRepository(String name, URI uri, String fileName) {
        super(name, uri);
        this.fileName = fileName;
        modelName = arguments.get("model_name");
        artifactId = arguments.get("artifact_id");
        if (artifactId == null) {
            artifactId = FilenameUtils.getNamePart(fileName);
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
        List<Artifact> artifacts = locate(mrl).getArtifacts();
        if (artifacts.isEmpty()) {
            return null;
        }
        return artifacts.get(0);
    }

    /** {@inheritDoc} */
    @Override
    public List<MRL> getResources() {
        try {
            Metadata m = getMetadata();
            if (m != null && !m.getArtifacts().isEmpty()) {
                MRL mrl = MRL.undefined(this, m.getGroupId(), m.getArtifactId());
                return Collections.singletonList(mrl);
            }
        } catch (IOException e) {
            logger.warn("Failed to connect URL: {}", uri, e);
        }
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    protected void download(Path tmp, URI baseUri, Artifact.Item item, Progress progress)
            throws IOException {
        logger.debug("Downloading artifact: {} ...", uri);
        try (InputStream is = new BufferedInputStream(uri.toURL().openStream())) {
            save(is, tmp, item, progress);
        }
    }

    private synchronized Metadata getMetadata() throws IOException {
        if (resolved) {
            return metadata;
        }

        Artifact artifact = new Artifact();
        artifact.setName(modelName);
        artifact.getArguments().putAll(arguments);
        Map<String, Artifact.Item> files = new ConcurrentHashMap<>();
        Artifact.Item item = new Artifact.Item();
        item.setUri(uri.getPath());
        item.setName(""); // avoid creating extra folder
        item.setExtension(FilenameUtils.getFileType(fileName));
        item.setArtifact(artifact);
        item.setSize(getContentLength());
        files.put(artifactId, item);
        artifact.setFiles(files);

        metadata = new Metadata.MatchAllMetadata();
        metadata.setArtifactId(artifactId);
        metadata.setArtifacts(Collections.singletonList(artifact));
        String hash = Utils.hash(uri.toString());
        MRL mrl = model(Application.UNDEFINED, DefaultModelZoo.GROUP_ID, hash);
        metadata.setRepositoryUri(mrl.toURI());
        return metadata;
    }

    private long getContentLength() throws IOException {
        String scheme = uri.getScheme();
        if ("http".equalsIgnoreCase(scheme) || "https".equalsIgnoreCase(scheme)) {
            HttpURLConnection conn = null;
            try {
                resolved = true;
                conn = (HttpURLConnection) uri.toURL().openConnection();
                conn.setRequestMethod("HEAD");
                int code = conn.getResponseCode();
                if (code != 200) {
                    logger.debug("Failed detect content length, error code: {}", code);
                    return -1;
                }
                return conn.getContentLength();
            } finally {
                if (conn != null) {
                    conn.disconnect();
                }
            }
        }
        return -1;
    }
}
