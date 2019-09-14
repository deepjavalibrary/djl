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
package software.amazon.ai.engine;

import java.lang.management.MemoryUsage;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.Device;
import software.amazon.ai.Model;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.ParameterStore;
import software.amazon.ai.training.optimizer.Optimizer;

/**
 * The {@code Engine} interface shadows differences between each deep learning framework.
 *
 * <p>Any framework specific functionality should be provided through this class.
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
            throw new EngineException("No deep learning engine found in class path.");
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
     * @return name of the engine
     */
    public abstract String getEngineName();

    /**
     * Returns the default Engine.
     *
     * @return instance of {@code Engine}
     */
    public static Engine getInstance() {
        return ENGINE;
    }

    /**
     * Returns {@code Engine} by engine name.
     *
     * @param engineName name of Engine to retrieve
     * @return instance of {@code Engine}
     */
    public static Engine getEngine(String engineName) {
        return ALL_ENGINES.get(engineName);
    }

    /**
     * Returns the number of GPUs available in the system.
     *
     * @return number of GPUs available in the system
     */
    public abstract int getGpuCount();

    /**
     * Returns {@link MemoryUsage} of specified GPU device.
     *
     * @param device the GPU {@link Device} to retrieve
     * @return {@link MemoryUsage} of specified GPU device
     * @throws EngineException if operation is not supported
     * @throws IllegalArgumentException if Device is not GPU device
     */
    public abstract MemoryUsage getGpuMemory(Device device);

    /**
     * Returns system default device.
     *
     * <p>If the system has GPU available, then the default device is {@link Device#gpu()}.
     * Otherwise the default device returned is {@link Device#cpu()}
     *
     * @return default device
     */
    public abstract Device defaultDevice();

    /**
     * Returns the version of the deep learning framework.
     *
     * @return version number
     */
    public abstract String getVersion();

    /**
     * Construct a new model.
     *
     * @param device the device that model to be loaded
     * @return a new Model instance using the network defined in block
     */
    public abstract Model newModel(Device device);

    /**
     * Creates a new {@link GradientCollector} instance for this Engine.
     *
     * @return Returns the GradientCollector
     */
    public abstract GradientCollector newGradientCollector();

    /**
     * An internal helper to get the Engine specific implementation for parameter store.
     *
     * @param optimizer The optimizer that defines how to update parameters
     * @param aggregateOnGPU whether to aggregate gradients on GPU for multi GPU training, if false,
     *     gradients will be copied to CPU for aggregation
     * @return {@link ParameterStore} object
     */
    public abstract ParameterStore newParameterStore(Optimizer optimizer, boolean aggregateOnGPU);

    /**
     * Creates a new top-level {@link NDManager}.
     *
     * <p>{@code NDManager} will inherit default {@link Device}.
     *
     * @return Returns a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager();

    /**
     * Creates a new top-level {@link NDManager} with specified {@link Device}.
     *
     * @param device default {@link Device}
     * @return Returns a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager(Device device);
}
