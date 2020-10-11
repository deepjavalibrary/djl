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
import ai.djl.training.LocalParameterServer;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.cuda.CudaUtils;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code Engine} interface is the base of the provided implementation for DJL.
 *
 * <p>Any engine-specific functionality should be provided through this class. In general, it should
 * contain methods to detect information about the usable machine hardware and to create a new
 * {@link NDManager} and {@link Model}.
 *
 * @see EngineProvider
 */
public abstract class Engine {

    private static final Logger logger = LoggerFactory.getLogger(Engine.class);

    private static final Map<String, Engine> ALL_ENGINES = new ConcurrentHashMap<>();

    private static final String DEFAULT_ENGINE = initEngine();

    private static EngineException exception;

    private static synchronized String initEngine() {
        ServiceLoader<EngineProvider> loaders = ServiceLoader.load(EngineProvider.class);
        for (EngineProvider provider : loaders) {
            try {
                Engine engine = provider.getEngine();
                logger.debug("Engine loaded from provider: {}", engine.getEngineName());
                ALL_ENGINES.put(engine.getEngineName(), engine);
            } catch (EngineException e) {
                exception = e;
                logger.warn("Failed to load engine from: " + provider.getClass().getName(), e);
            }
        }

        if (ALL_ENGINES.isEmpty()) {
            logger.debug("No engine found from EngineProvider");
            return null;
        }

        String defaultEngine = System.getenv("DJL_DEFAULT_ENGINE");
        defaultEngine = System.getProperty("ai.djl.default_engine", defaultEngine);
        if (defaultEngine == null || defaultEngine.isEmpty()) {
            if (ALL_ENGINES.size() > 1) {
                logger.warn("More than one deep learning engines found.");
            }
            int rank = Integer.MAX_VALUE;
            for (Engine engine : ALL_ENGINES.values()) {
                if (engine.getRank() < rank) {
                    defaultEngine = engine.getEngineName();
                    rank = engine.getRank();
                }
            }
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
     * Return the rank of the {@code Engine}.
     *
     * @return the rank of the engine
     */
    public int getRank() {
        return 1;
    }

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
                            + "Please refer to https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md for more details.",
                    exception);
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
     * Returns a set of engine names that are loaded.
     *
     * @return a set of engine names that are loaded
     */
    public static Set<String> getAllEngines() {
        return ALL_ENGINES.keySet();
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
     * Returns the version of the deep learning engine.
     *
     * @return the version number of the deep learning engine
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
     * Returns a new instance of {@link ParameterServer}.
     *
     * @param optimizer the optimizer to update
     * @return a new instance of {@link ParameterServer}
     */
    public ParameterServer newParameterServer(Optimizer optimizer) {
        return new LocalParameterServer(optimizer);
    }

    /**
     * Seeds the random number generator in DJL Engine.
     *
     * <p>This will affect all {@link Device}s and all operators using Engine's random number
     * generator.
     *
     * @param seed the seed to be fixed in Engine
     */
    public abstract void setRandomSeed(int seed);

    /** Prints debug information about the environment for debugging environment issues. */
    @SuppressWarnings("PMD.SystemPrintln")
    public static void debugEnvironment() {
        System.out.println("----------- System Properties -----------");
        System.getProperties().forEach((k, v) -> System.out.println(k + ": " + v));

        System.out.println();
        System.out.println("--------- Environment Variables ---------");
        System.getenv().forEach((k, v) -> System.out.println(k + ": " + v));

        System.out.println();
        System.out.println("-------------- Directories --------------");
        try {
            Path temp = Paths.get(System.getProperty("java.io.tmpdir"));
            System.out.println("temp directory: " + temp.toString());
            Files.createTempFile("test", ".tmp");

            Path path = getEngineCacheDir();
            System.out.println("Engine cache directory: " + path.toAbsolutePath().toString());
            Files.createDirectories(path);
            if (!Files.isWritable(path)) {
                System.out.println("Engine cache directory is not writable!!!");
            }
        } catch (Throwable e) {
            e.printStackTrace(System.out);
        }

        System.out.println();
        System.out.println("------------------ CUDA -----------------");
        int gpuCount = Device.getGpuCount();
        System.out.println("GPU Count: " + gpuCount);
        System.out.println("Default Device: " + Device.defaultDevice());
        if (gpuCount > 0) {
            System.out.println("CUDA: " + CudaUtils.getCudaVersionString());
            System.out.println("ARCH: " + CudaUtils.getComputeCapability(0));
        }
        for (int i = 0; i < gpuCount; ++i) {
            Device device = Device.gpu(i);
            MemoryUsage mem = CudaUtils.getGpuMemory(device);
            System.out.println("GPU(" + i + ") memory used: " + mem.getCommitted() + " bytes");
        }

        System.out.println();
        System.out.println("----------------- Engines ---------------");
        System.out.println("Default Engine: " + DEFAULT_ENGINE);
        for (Engine engine : ALL_ENGINES.values()) {
            System.out.println(engine);
        }
        if (exception != null) {
            System.out.println("Last error:");
            exception.printStackTrace(System.out);
        }
    }

    private static Path getEngineCacheDir() {
        String cacheDir = System.getProperty("ENGINE_CACHE_DIR");
        if (cacheDir == null || cacheDir.isEmpty()) {
            cacheDir = System.getenv("ENGINE_CACHE_DIR");
            if (cacheDir == null || cacheDir.isEmpty()) {
                cacheDir = System.getProperty("DJL_CACHE_DIR");
                if (cacheDir == null || cacheDir.isEmpty()) {
                    cacheDir = System.getenv("DJL_CACHE_DIR");
                    if (cacheDir == null || cacheDir.isEmpty()) {
                        String userHome = System.getProperty("user.home");
                        return Paths.get(userHome, ".djl.ai");
                    }
                }
            }
        }
        return Paths.get(cacheDir);
    }
}
