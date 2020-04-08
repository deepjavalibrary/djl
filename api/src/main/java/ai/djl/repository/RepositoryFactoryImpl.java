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

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class RepositoryFactoryImpl implements RepositoryFactory {

    private static final Logger logger = LoggerFactory.getLogger(RepositoryFactoryImpl.class);

    private static final RepositoryFactory FACTORY = new RepositoryFactoryImpl();

    private static final Map<String, RepositoryFactory> REGISTRY = init();

    static RepositoryFactory getFactory() {
        return FACTORY;
    }

    /** {@inheritDoc} */
    @Override
    public Repository newInstance(String name, String url) {
        URI uri = URI.create(url);
        String scheme = uri.getScheme();
        if (scheme == null) {
            scheme = "file";
        }
        RepositoryFactory factory = REGISTRY.get(scheme);
        if (factory != null) {
            return factory.newInstance(name, url);
        }

        if ("file".equalsIgnoreCase(scheme)) {
            Path path = Paths.get(uri.getPath());
            if (Files.exists(path) && Files.isDirectory(path)) {
                try {
                    if (Files.walk(path)
                            .anyMatch(f -> f.endsWith("metadata.json") && Files.isRegularFile(f))) {
                        return new LocalRepository(name, path);
                    }
                } catch (IOException e) {
                    logger.warn("Failed locate metadata.json file, defaulting to simple", e);
                }
            }
            return new SimpleRepository(name, path);
        }
        return new RemoteRepository(name, uri);
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedScheme() {
        return Collections.emptySet();
    }

    static void registerRepositoryFactory(RepositoryFactory factory) {
        for (String scheme : factory.getSupportedScheme()) {
            REGISTRY.put(scheme, factory);
        }
    }

    private static Map<String, RepositoryFactory> init() {
        Map<String, RepositoryFactory> registry = new ConcurrentHashMap<>();
        ServiceLoader<RepositoryFactory> factories = ServiceLoader.load(RepositoryFactory.class);
        for (RepositoryFactory factory : factories) {
            for (String scheme : factory.getSupportedScheme()) {
                registry.put(scheme, factory);
            }
        }
        return registry;
    }
}
