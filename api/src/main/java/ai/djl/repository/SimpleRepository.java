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

import ai.djl.repository.Artifact.Item;
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.util.Progress;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
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

    private String name;
    private Path path;

    /**
     * (Internal) Constructs a SimpleRepository.
     *
     * <p>Use {@link Repository#newInstance(String, String)}.
     *
     * @param name the name of the repository
     * @param path the path to the repository
     */
    protected SimpleRepository(String name, Path path) {
        this.name = name;
        this.path = path;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isRemote() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return name;
    }

    /** {@inheritDoc} */
    @Override
    public URI getBaseUri() {
        return path.toUri();
    }

    /** {@inheritDoc} */
    @Override
    public Metadata locate(MRL mrl) throws IOException {
        Path file = path.resolve("metadata.json");
        if (Files.isRegularFile(file)) {
            logger.debug("Using metadata.json file: {}", file.toAbsolutePath());
            return metadataWithFile(file);
        }
        logger.debug("No metadata.json file found in: {}", path.toAbsolutePath());
        return metadataWithoutFile();
    }

    private Metadata metadataWithFile(Path file) throws IOException {
        try (Reader reader = Files.newBufferedReader(file)) {
            Metadata metadata = GSON.fromJson(reader, Metadata.class);
            metadata.setRepositoryUri(URI.create(""));
            return metadata;
        }
    }

    private Metadata metadataWithoutFile() {
        Metadata metadata = new Metadata.MatchAllMetadata();
        metadata.setRepositoryUri(URI.create(""));
        File file = path.toFile();
        if (Files.isRegularFile(path)) {
            metadata.setArtifactId(file.getParentFile().getName());
        } else {
            metadata.setArtifactId(file.getName());
        }
        if (!Files.exists(path)) {
            logger.debug("Specified path doesn't exists: {}", path.toAbsolutePath());
            return metadata;
        }

        Artifact artifact = new Artifact();
        artifact.setName(file.getName());
        Map<String, Item> files = new ConcurrentHashMap<>();
        if (file.isDirectory()) {
            File[] fileList = file.listFiles();
            if (fileList != null) {
                for (File f : fileList) {
                    Item item = new Item();
                    item.setName(f.getName());
                    item.setSize(f.length());
                    item.setArtifact(artifact);
                    files.put(f.getName(), item);
                }
            }
        }
        artifact.setFiles(files);

        metadata.setArtifacts(Collections.singletonList(artifact));
        return metadata;
    }

    /** {@inheritDoc} */
    @Override
    public Artifact resolve(MRL mrl, String version, Map<String, String> filter)
            throws IOException {
        List<Artifact> artifacts = locate(mrl).getArtifacts();
        if (artifacts.isEmpty()) {
            return null;
        }
        return artifacts.get(0);
    }

    /** {@inheritDoc} */
    @Override
    public Path getResourceDirectory(Artifact artifact) {
        return path;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Artifact artifact, Progress progress) {
        // Do nothing
    }

    /** {@inheritDoc} */
    @Override
    public Path getCacheDirectory() {
        return path;
    }

    /** {@inheritDoc} */
    @Override
    protected URI resolvePath(Item item, String path) {
        return this.path.resolve(item.getName()).toUri();
    }

    /** {@inheritDoc} */
    @Override
    public List<MRL> getResources() {
        if (!Files.exists(path)) {
            logger.debug("Specified path doesn't exists: {}", path.toAbsolutePath());
            return Collections.emptyList();
        }

        MRL mrl = MRL.undefined(DefaultModelZoo.GROUP_ID, path.toFile().getName());
        return Collections.singletonList(mrl);
    }
}
