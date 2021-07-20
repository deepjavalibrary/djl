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
import ai.djl.util.JsonUtils;
import java.io.IOException;
import java.io.Reader;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A {@code LocalRepository} is a {@link Repository} located in a filesystem directory.
 *
 * @see Repository
 */
public class LocalRepository extends AbstractRepository {

    private static final Logger logger = LoggerFactory.getLogger(LocalRepository.class);

    private String name;
    private Path path;

    /**
     * (Internal) Constructs a {@code LocalRepository} from the path with inferred name.
     *
     * <p>Use {@link Repository#newInstance(String, String)}.
     *
     * @param name the name of the repository
     * @param path the path to the repository
     */
    protected LocalRepository(String name, Path path) {
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
        URI uri = mrl.toURI();
        Path base = path.resolve(uri.getPath());
        Path file = base.resolve("metadata.json");
        if (!Files.isRegularFile(file)) {
            return null;
        }
        try (Reader reader = Files.newBufferedReader(file)) {
            Metadata metadata = JsonUtils.GSON_PRETTY.fromJson(reader, Metadata.class);
            metadata.init();
            metadata.setRepositoryUri(uri);
            return metadata;
        }
    }

    /** {@inheritDoc} */
    @Override
    public Artifact resolve(MRL mrl, Map<String, String> filter) throws IOException {
        Metadata metadata = locate(mrl);
        VersionRange range = VersionRange.parse(mrl.getVersion());
        List<Artifact> artifacts = metadata.search(range, filter);
        if (artifacts.isEmpty()) {
            return null;
        }
        // TODO: find highest version.
        return artifacts.get(0);
    }

    /** {@inheritDoc} */
    @Override
    public List<MRL> getResources() {
        List<MRL> list = new ArrayList<>();
        try {
            Files.walk(path)
                    .forEach(
                            f -> {
                                if (f.endsWith("metadata.json") && Files.isRegularFile(f)) {
                                    Path relative = path.relativize(f);
                                    String type = relative.getName(0).toString();
                                    try (Reader reader = Files.newBufferedReader(f)) {
                                        Metadata metadata =
                                                JsonUtils.GSON.fromJson(reader, Metadata.class);
                                        Application application = metadata.getApplication();
                                        String groupId = metadata.getGroupId();
                                        String artifactId = metadata.getArtifactId();
                                        if ("dataset".equals(type)) {
                                            list.add(dataset(application, groupId, artifactId));
                                        } else if ("model".equals(type)) {
                                            list.add(model(application, groupId, artifactId));
                                        }
                                    } catch (IOException e) {
                                        logger.warn("Failed to read metadata.json", e);
                                    }
                                }
                            });
        } catch (IOException e) {
            logger.warn("", e);
        }
        return list;
    }
}
