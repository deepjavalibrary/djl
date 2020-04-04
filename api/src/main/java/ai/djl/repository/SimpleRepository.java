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
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@code SimpleRepository} is a {@link Repository} containing only a single artifact without
 * requiring a "metadata.json" file.
 *
 * @see Repository
 */
public class SimpleRepository extends AbstractRepository {

    private static final String GROUP_ID = "ai.djl.localmodelzoo";

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
    public Metadata locate(MRL mrl) {
        File file = path.toFile();
        Metadata metadata = new MatchAllMetadata();
        metadata.setGroupId(GROUP_ID);
        if (Files.isRegularFile(path)) {
            metadata.setArtifactId(file.getParentFile().getName());
        } else {
            metadata.setArtifactId(file.getName());
        }
        metadata.setRepositoryUri(URI.create(""));
        if (!Files.exists(path)) {
            metadata.setArtifacts(Collections.emptyList());
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
                artifact.setFiles(files);
            }
        }
        artifact.setFiles(files);

        metadata.setArtifacts(Collections.singletonList(artifact));
        return metadata;
    }

    /** {@inheritDoc} */
    @Override
    public Artifact resolve(MRL mrl, String version, Map<String, String> filter) {
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
    protected URI resolvePath(Item item, String path) {
        return this.path.resolve(item.getName()).toUri();
    }

    /** {@inheritDoc} */
    @Override
    public List<MRL> getResources() {
        if (!Files.exists(path)) {
            return Collections.emptyList();
        }

        MRL mrl = new MRL(new Anchor(), GROUP_ID, path.toFile().getName());
        return Collections.singletonList(mrl);
    }

    private static final class MatchAllMetadata extends Metadata {

        /** {@inheritDoc} */
        @Override
        public List<Artifact> search(VersionRange versionRange, Map<String, String> filter) {
            return getArtifacts();
        }
    }
}
