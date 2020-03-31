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
import ai.djl.util.Progress;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@code SimpleRepository} is a {@link Repository} containing only a single artifact without
 * requiring a "metadata.json" file.
 *
 * @see Repository
 */
public class SimpleRepository extends AbstractRepository {

    private String name;
    private Path path;

    /**
     * (Internal) Constructs a SimpleRepository.
     *
     * <p>Use {@link Repository#newInstance(String, String)}.
     *
     * @param path the path to the repository
     */
    public SimpleRepository(Path path) {
        this(path.toFile().getName(), path);
    }

    /**
     * (Internal) Constructs a SimpleRepository.
     *
     * <p>Use {@link Repository#newInstance(String, String)}.
     *
     * @param name the name of the repository
     * @param path the path to the repository
     */
    public SimpleRepository(String name, Path path) {
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
        Metadata metadata = new Metadata();
        metadata.setRepositoryUri(URI.create(""));
        Artifact artifact = new Artifact();
        artifact.setMetadata(metadata);
        metadata.setArtifacts(Collections.singletonList(artifact));
        artifact.setName(name);
        Map<String, Item> files = new ConcurrentHashMap<>();
        File[] fileList = path.toFile().listFiles();
        if (fileList == null) {
            throw new IllegalArgumentException("No files found in SimpleRepository");
        }
        for (File file : fileList) {
            Item item = new Item();
            item.setName(file.getName());
            item.setSize(file.length());
            item.setArtifact(artifact);
            files.put(file.getName(), item);
        }
        artifact.setFiles(files);
        return metadata;
    }

    /** {@inheritDoc} */
    @Override
    public Artifact resolve(MRL mrl, String version, Map<String, String> filter)
            throws IOException {
        return locate(mrl).getArtifacts().get(0);
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
    protected URI resolvePath(Item item, String path) throws IOException {
        return this.path.resolve(item.getName()).toUri();
    }
}
