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
import java.lang.management.MemoryUsage;
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

    private static final Engine ENGINE = initEngine();

    private static synchronized Engine initEngine() {
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

        if (list.size() > 1) {
            logger.warn("More than one deep learning engines found.");
        }

        Engine engine = list.get(0).getEngine();
        logger.debug("Loading ML engine from: {}", engine.getClass());
        return engine;
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
        if (ENGINE == null) {
            throw new EngineException("No deep learning engine found in class path.");
        }
        return ENGINE;
    }

    /**
     * Returns the {@code Engine} with the given name.
     *
     * @param engineName the name of Engine to retrieve
     * @return the instance of {@code Engine}
     * @see EngineProvider
     */
    public static Engine getEngine(String engineName) {
        return ALL_ENGINES.get(engineName);
    }

    /**
     * Returns the number of GPUs available in the system.
     *
     * @return the number of GPUs available in the system
     */
    public abstract int getGpuCount();

    /**
     * Returns the {@link MemoryUsage} of the specified GPU device.
     *
     * @param device the GPU {@link Device} to retrieve
     * @return the {@link MemoryUsage} of the specified GPU device
     * @throws EngineException if operation is not supported
     * @throws IllegalArgumentException if {@link Device} is not GPU device
     */
    public abstract MemoryUsage getGpuMemory(Device device);

    /**
     * Returns the system's default device.
     *
     * <p>If the system has GPU available, then the default device is {@link Device#gpu()}.
     * Otherwise, the default device returned is {@link Device#cpu()}
     *
     * @return the system's default device
     */
    public abstract Device defaultDevice();

    /**
     * Returns the version of the deep learning framework.
     *
     * @return the version number of the deep learning framework
     */
    public abstract String getVersion();

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
}
