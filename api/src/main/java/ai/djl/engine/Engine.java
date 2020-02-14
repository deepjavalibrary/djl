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
package ai.djl.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code Engine} interface is the base of the provided implementation for DJL.
 *
 * <p>Any framework-specific functionality should be provided through this class. In general, it
 * should contain methods to detect information about the usable machine hardware and to create a
 * new {@link NDManager} and {@link Model}.
 *
 * @see EngineProvider
 */
public abstract class Engine {

    private static final Logger logger = LoggerFactory.getLogger(Engine.class);

    private static final Map<String, Engine> ALL_ENGINES = new ConcurrentHashMap<>();

    private static final String DEFAULT_ENGINE = initEngine();

    private static synchronized String initEngine() {
        ServiceLoader<EngineProvider> loaders = ServiceLoader.load(EngineProvider.class);
        List<EngineProvider> list = new ArrayList<>();
        for (EngineProvider provider : loaders) {
            list.add(provider);
            Engine engine = provider.getEngine();
            ALL_ENGINES.put(engine.getEngineName(), engine);
        }

        if (list.isEmpty()) {
            return null;
        }

        Engine engine = list.get(0).getEngine();
        String defaultEngine = System.getenv("DJL_DEFAULT_ENGINE");
        if (defaultEngine == null || defaultEngine.isEmpty()) {
            defaultEngine = System.getProperty("ai.djl.default_engine");
        }
        if (defaultEngine == null || defaultEngine.isEmpty()) {
            if (list.size() > 1) {
                logger.warn("More than one deep learning engines found.");
            }
            defaultEngine = engine.getEngineName();
        } else if (!ALL_ENGINES.containsKey(defaultEngine)) {
            throw new EngineException("Unknown default engine: " + defaultEngine);
        }
        logger.debug("Found default engine: {}", defaultEngine);
        return defaultEngine;
    }

    /**
     * Returns the name of the Engine.
     *
     * @return the name of the engine
     */
    public abstract String getEngineName();

    /**
     * Returns the default Engine.
     *
     * @return the instance of {@code Engine}
     * @see EngineProvider
     */
    public static Engine getInstance() {
        if (DEFAULT_ENGINE == null) {
            throw new EngineException("No deep learning engine found in class path.");
        }
        return getEngine(DEFAULT_ENGINE);
    }

    /**
     * Returns the {@code Engine} with the given name.
     *
     * @param engineName the name of Engine to retrieve
     * @return the instance of {@code Engine}
     * @see EngineProvider
     */
    public static Engine getEngine(String engineName) {
        Engine engine = ALL_ENGINES.get(engineName);
        if (engine == null) {
            throw new IllegalArgumentException("Deep learning engine not found: " + engineName);
        }
        return engine;
    }

    /**
     * Returns the version of the deep learning framework.
     *
     * @return the version number of the deep learning framework
     */
    public abstract String getVersion();

    /**
     * Returns whether the engine has the specified capability.
     *
     * @param capability the capability to retrieve
     * @return {@code true} if the engine has the specified capability
     */
    public abstract boolean hasCapability(String capability);

    /**
     * Constructs a new model.
     *
     * @param device the device that the model will be loaded onto
     * @return a new Model instance using the network defined in block
     */
    public abstract Model newModel(Device device);

    /**
     * Creates a new top-level {@link NDManager}.
     *
     * <p>{@code NDManager} will inherit default {@link Device}.
     *
     * @return a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager();

    /**
     * Creates a new top-level {@link NDManager} with specified {@link Device}.
     *
     * @param device the default {@link Device}
     * @return a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager(Device device);

    /**
     * Seeds the random number generator in DJL Engine.
     *
     * <p>This will affect all {@link Device}s and all operators using Engine's random number
     * generator.
     *
     * @param seed the seed to be fixed in Engine
     */
    public abstract void setRandomSeed(int seed);

    /** Logs debug information about the environment for use when debugging environment issues. */
    public void debugEnvironment() {
        logger.info("Engine name: {}", getEngineName());
        logger.info("Engine version: {}", getVersion());
    }
}
