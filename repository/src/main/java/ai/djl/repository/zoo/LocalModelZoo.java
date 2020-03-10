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
package ai.djl.repository.zoo;

import ai.djl.engine.Engine;
import ai.djl.repository.SimpleRepository;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A {@link ModelZoo} that contains models in local directory. */
public class LocalModelZoo implements ModelZoo {

    private static final Logger logger = LoggerFactory.getLogger(LocalModelZoo.class);

    public static final String GROUP_ID = "ai.djl.localmodelzoo";

    private Path folder;

    /**
     * Creates the {@code LocalModelZoo} instance from the given directory.
     *
     * @param folder the directory to load models from
     */
    public LocalModelZoo(Path folder) {
        this.folder = folder;
    }

    /** {@inheritDoc} */
    @Override
    public List<ModelLoader<?, ?>> getModelLoaders() {
        try {
            List<Path> dirs =
                    Files.list(folder)
                            .filter(p -> Files.isDirectory(p))
                            .collect(Collectors.toList());
            if (dirs.isEmpty()) {
                LocalModelLoader loader = new LocalModelLoader(new SimpleRepository(folder));
                return Collections.singletonList(loader);
            }

            List<ModelLoader<?, ?>> loaders = new ArrayList<>();
            for (Path p : dirs) {
                loaders.add(new LocalModelLoader(new SimpleRepository(p)));
            }
            return loaders;
        } catch (IOException e) {
            logger.error("Failed list files.", e);
        }
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    public String getGroupId() {
        return GROUP_ID;
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedEngines() {
        return Collections.singleton(Engine.getInstance().getEngineName());
    }
}
