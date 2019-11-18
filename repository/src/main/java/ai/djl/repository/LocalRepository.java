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

import java.io.IOException;
import java.io.Reader;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * A {@code LocalRepository} is a {@link Repository} located in a filesystem directory.
 *
 * @see Repository
 */
public class LocalRepository extends AbstractRepository {

    private String name;
    private Path path;

    /**
     * (Internal) Constructs a {@code LocalRepository} from the path with inferred name.
     *
     * <p>Use {@link Repository#newInstance(String, String)}.
     *
     * @param path the path to the repository
     */
    public LocalRepository(Path path) {
        this(path.toFile().getName(), path);
    }

    /**
     * (Internal) Constructs a {@code LocalRepository} from the path with inferred name.
     *
     * <p>Use {@link Repository#newInstance(String, String)}.
     *
     * @param name the name of the repository
     * @param path the path to the repository
     */
    public LocalRepository(String name, Path path) {
        this.name = name;
        this.path = path;
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
        URI uri = mrl.toURI();
        Path base = path.resolve(uri.getPath());
        Path file = base.resolve("metadata.json");
        if (!Files.isRegularFile(file)) {
            return null;
        }
        try (Reader reader = Files.newBufferedReader(file)) {
            Metadata metadata = GSON.fromJson(reader, Metadata.class);
            metadata.setRepositoryUri(uri);
            return metadata;
        }
    }

    /** {@inheritDoc} */
    @Override
    public Artifact resolve(MRL mrl, String version, Map<String, String> filter)
            throws IOException {
        Metadata metadata = locate(mrl);
        VersionRange range = VersionRange.parse(version);
        List<Artifact> artifacts = metadata.search(range, filter);
        if (artifacts.isEmpty()) {
            return null;
        }
        // TODO: find highest version.
        return artifacts.get(0);
    }
}
