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
package ai.djl.hadoop.hdfs;

import ai.djl.Application;
import ai.djl.repository.AbstractRepository;
import ai.djl.repository.Artifact;
import ai.djl.repository.FilenameUtils;
import ai.djl.repository.MRL;
import ai.djl.repository.Metadata;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.util.Progress;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A {@code HdfsRepository} is a {@link Repository} located on a Hadoop HDFS.
 *
 * @see Repository
 */
public class HdfsRepository extends AbstractRepository {

    private static final Logger logger = LoggerFactory.getLogger(HdfsRepository.class);

    private Configuration config;
    private String prefix;
    private String artifactId;
    private String modelName;

    private Metadata metadata;
    private boolean resolved;
    private boolean isDirectory;

    HdfsRepository(String name, URI uri, Configuration config) {
        super(name, uri);
        this.config = config;

        prefix = uri.getPath();
        String fileName = Paths.get(prefix).toFile().getName();
        isDirectory = !FilenameUtils.isArchiveFile(fileName);
        if (!isDirectory) {
            fileName = FilenameUtils.getNamePart(fileName);
        }

        modelName = arguments.get("model_name");
        artifactId = arguments.get("artifact_id");
        if (artifactId == null) {
            artifactId = fileName;
        }
        if (modelName == null) {
            modelName = artifactId;
        }
        if (prefix.isEmpty()) {
            prefix = "/";
        }

        try {
            this.uri =
                    new URI(
                            uri.getScheme(),
                            uri.getUserInfo(),
                            uri.getHost(),
                            uri.getPort(),
                            null,
                            null,
                            null);
        } catch (URISyntaxException e) {
            throw new AssertionError(e);
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
        FileSystem fs = FileSystem.get(uri, config);
        org.apache.hadoop.fs.Path path = new org.apache.hadoop.fs.Path(item.getUri());
        logger.debug("Downloading artifact: {} ...", path);
        try (InputStream is = fs.open(path)) {
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
            logger.warn("Failed to scan: " + uri.toString(), e);
        }
        return Collections.emptyList();
    }

    private synchronized Metadata getMetadata() throws IOException {
        if (resolved) {
            return metadata;
        }

        resolved = true;
        Artifact artifact = listFiles();
        if (artifact == null) {
            logger.debug("No object found in hdfs: " + uri);
            return null;
        }

        metadata = new Metadata.MatchAllMetadata();
        String hash = md5hash(uri.resolve(prefix).toString());
        MRL mrl = model(Application.UNDEFINED, DefaultModelZoo.GROUP_ID, hash);
        metadata.setRepositoryUri(mrl.toURI());
        metadata.setArtifactId(artifactId);
        metadata.setArtifacts(Collections.singletonList(artifact));
        return metadata;
    }

    private Artifact listFiles() throws IOException {
        FileSystem fs = FileSystem.get(uri, config);
        FileStatus[] status = fs.listStatus(new org.apache.hadoop.fs.Path(prefix));
        if (status == null || status.length == 0) {
            return null;
        }

        Artifact artifact = new Artifact();
        artifact.setName(modelName);
        artifact.getArguments().putAll(arguments);
        Map<String, Artifact.Item> files = new ConcurrentHashMap<>();
        artifact.setFiles(files);
        if (isDirectory) {
            Path fullPath = Paths.get(prefix);
            for (FileStatus st : status) {
                Artifact.Item item = new Artifact.Item();
                String key = st.getPath().getName();
                if (!key.endsWith("/")) {
                    item.setUri(fullPath.resolve(key).toString());
                    item.setSize(st.getLen());
                    item.setArtifact(artifact);
                    if ("dir".equals(item.getType())) {
                        item.setName(""); // avoid creating extra folder
                    }
                    files.put(key, item);
                }
            }
        } else {
            Artifact.Item item = new Artifact.Item();
            item.setUri(prefix);
            item.setName(""); // avoid creating extra folder
            item.setArtifact(artifact);
            item.setSize(status[0].getLen());
            files.put(artifactId, item);
        }
        return artifact;
    }
}
