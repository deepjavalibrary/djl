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
import ai.djl.training.GradientCollector;
import java.util.Collection;
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
        Engine firstEngine = null;
        ServiceLoader<EngineProvider> loaders = ServiceLoader.load(EngineProvider.class);
        for (EngineProvider provider : loaders) {
            Engine engine = provider.getEngine();
            if (engine != null) {
                logger.debug("Engine loaded from provider: {}", engine.getEngineName());
                if (firstEngine == null) {
                    firstEngine = engine;
                }
                ALL_ENGINES.put(engine.getEngineName(), engine);
            } else {
                logger.warn("Failed to load engine from: {}", provider.getClass().getName());
            }
        }

        if (firstEngine == null) {
            logger.debug("No engine found from EngineProvider");
            return null;
        }

        String defaultEngine = System.getenv("DJL_DEFAULT_ENGINE");
        defaultEngine = System.getProperty("ai.djl.default_engine", defaultEngine);
        if (defaultEngine == null || defaultEngine.isEmpty()) {
            if (ALL_ENGINES.size() > 1) {
                logger.warn("More than one deep learning engines found.");
            }
            defaultEngine = firstEngine.getEngineName();
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
            throw new EngineException(
                    "No deep learning engine found."
                            + System.lineSeparator()
                            + "Please refer to https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md for more details.");
        }
        return getEngine(System.getProperty("ai.djl.default_engine", DEFAULT_ENGINE));
    }

    /**
     * Returns if the specified engine is available.
     *
     * @param engineName the name of Engine to check
     * @return {@code true} if the specified engine is available
     * @see EngineProvider
     */
    public static boolean hasEngine(String engineName) {
        return ALL_ENGINES.containsKey(engineName);
    }

    /**
     * Returns a Collection of engines that are loaded.
     *
     * @return {@code Collection<Engine>} that are supported
     */
    public static Collection<Engine> getAllEngines() {
        return ALL_ENGINES.values();
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
     * @param name the model name
     * @param device the device that the model will be loaded onto
     * @return a new Model instance using the network defined in block
     */
    public abstract Model newModel(String name, Device device);

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
     * Returns a new instance of {@link GradientCollector}.
     *
     * @return a new instance of {@link GradientCollector}
     */
    public abstract GradientCollector newGradientCollector();

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
