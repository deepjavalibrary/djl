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

import ai.djl.Application;
import ai.djl.repository.Artifact.Item;
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.util.Progress;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A {@code SimpleRepository} is a {@link Repository} containing only a single artifact without
 * requiring a "metadata.json" file.
 *
 * @see Repository
 */
public class SimpleRepository extends AbstractRepository {

    private static final Logger logger = LoggerFactory.getLogger(SimpleRepository.class);

    private Path path;
    private String artifactId;
    private String modelName;
    private boolean isRemote;

    private Metadata metadata;
    private boolean resolved;

    /**
     * (Internal) Constructs a SimpleRepository.
     *
     * <p>Use {@link Repository#newInstance(String, String)}.
     *
     * @param name the name of the repository
     * @param uri the base URI of the repository
     * @param path the path to the repository
     */
    protected SimpleRepository(String name, URI uri, Path path) {
        super(name, uri);
        this.path = path;
        isRemote = FilenameUtils.isArchiveFile(path.toString());
        modelName = arguments.get("model_name");
        artifactId = arguments.get("artifact_id");
        if (artifactId == null) {
            if (isRemote) {
                artifactId = FilenameUtils.getNamePart(path.toFile().getName());
            } else {
                artifactId = path.toFile().getName();
            }
        }
        if (modelName == null) {
            modelName = artifactId;
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean isRemote() {
        return isRemote;
    }

    /** {@inheritDoc} */
    @Override
    public URI getBaseUri() {
        return path.toUri();
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
    public Path getResourceDirectory(Artifact artifact) throws IOException {
        if (isRemote) {
            return super.getResourceDirectory(artifact);
        }
        return path;
    }

    /** {@inheritDoc} */
    @Override
    protected void download(Path tmp, URI baseUri, Artifact.Item item, Progress progress)
            throws IOException {
        logger.debug("Extracting artifact: {} ...", path);
        try (InputStream is = new BufferedInputStream(Files.newInputStream(path))) {
            save(is, tmp, item, progress);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Artifact artifact, Progress progress) throws IOException {
        if (isRemote) {
            super.prepare(artifact, progress);
        } else {
            logger.debug("Skip prepare for local repository.");
        }
    }

    /** {@inheritDoc} */
    @Override
    public Path getCacheDirectory() throws IOException {
        if (isRemote) {
            return super.getCacheDirectory();
        }
        return path;
    }

    /** {@inheritDoc} */
    @Override
    protected URI resolvePath(Item item, String path) throws IOException {
        if (isRemote) {
            return super.resolvePath(item, path);
        }
        return this.path.resolve(item.getName()).toUri();
    }

    /** {@inheritDoc} */
    @Override
    public List<MRL> getResources() {
        if (!Files.exists(path)) {
            logger.debug("Specified path doesn't exists: {}", path.toAbsolutePath());
            return Collections.emptyList();
        }

        MRL mrl = MRL.undefined(this, DefaultModelZoo.GROUP_ID, artifactId);
        return Collections.singletonList(mrl);
    }

    private synchronized Metadata getMetadata() throws IOException {
        if (resolved) {
            return metadata;
        }

        resolved = true;
        metadata = new Metadata.MatchAllMetadata();
        metadata.setRepositoryUri(URI.create(""));
        metadata.setArtifactId(artifactId);
        if (!Files.exists(path)) {
            logger.debug("Specified path doesn't exists: {}", path.toAbsolutePath());
            return metadata;
        }

        Artifact artifact = new Artifact();
        artifact.setName(modelName);
        artifact.getArguments().putAll(arguments);
        Map<String, Item> files = new ConcurrentHashMap<>();
        if (isRemote) {
            Artifact.Item item = new Artifact.Item();
            String uri = path.toAbsolutePath().toUri().toString();
            item.setUri(uri);
            item.setName(""); // avoid creating extra folder
            item.setArtifact(artifact);
            item.setSize(Files.size(path));
            files.put(artifactId, item);
            artifact.setFiles(files);

            String hash = md5hash(uri);
            MRL mrl = model(Application.UNDEFINED, DefaultModelZoo.GROUP_ID, hash);
            metadata.setRepositoryUri(mrl.toURI());
        } else {
            if (Files.isDirectory(path)) {
                File[] fileList = path.toFile().listFiles();
                if (fileList != null) {
                    for (File f : fileList) {
                        Item item = new Item();
                        item.setName(f.getName());
                        item.setSize(f.length());
                        item.setArtifact(artifact);
                        files.put(f.getName(), item);
                    }
                }
            } else {
                logger.warn("Simple repository pointing to a non-archive file.");
            }
        }
        artifact.setFiles(files);

        metadata.setArtifacts(Collections.singletonList(artifact));
        return metadata;
    }
}
