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
package ai.djl.mxnet.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.mxnet.jna.LibUtils;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.GradientCollector;
import ai.djl.training.LocalParameterServer;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.RandomUtils;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * The {@code MxEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://mxnet.apache.org/">Apache MXNet Deep Learning Framework</a>.
 *
 * <p>To get an instance of the {@code MxEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "MXNet".
 */
public final class MxEngine extends Engine {

    public static final String ENGINE_NAME = "MXNet";
    static final int RANK = 1;
    private static final String MXNET_EXTRA_LIBRARY_VERBOSE = "MXNET_EXTRA_LIBRARY_VERBOSE";

    /** Constructs an MXNet Engine. */
    private MxEngine() {}

    static Engine newInstance() {
        try {
            // Workaround MXNet engine lazy initialization issue
            JnaUtils.getAllOpNames();

            JnaUtils.setNumpyMode(JnaUtils.NumpyMode.GLOBAL_ON);

            // Workaround MXNet shutdown crash issue
            Runtime.getRuntime().addShutdownHook(new Thread(JnaUtils::waitAll)); // NOPMD

            // load extra MXNet library
            String paths = System.getenv("MXNET_EXTRA_LIBRARY_PATH");
            boolean extraLibVerbose = false;
            if (System.getenv().containsKey(MXNET_EXTRA_LIBRARY_VERBOSE)) {
                extraLibVerbose = Boolean.parseBoolean(System.getenv(MXNET_EXTRA_LIBRARY_VERBOSE));
            }
            if (paths != null) {
                String[] files = paths.split(",");
                for (String file : files) {
                    Path path = Paths.get(file);
                    if (Files.notExists(path)) {
                        throw new FileNotFoundException("Extra Library not found: " + file);
                    }
                    JnaUtils.loadLib(path.toAbsolutePath().toString(), extraLibVerbose);
                }
            }

            return new MxEngine();
        } catch (Throwable t) {
            throw new EngineException("Failed to load MXNet native library", t);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return RANK;
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        int version = JnaUtils.getVersion();
        int major = version / 10000;
        int minor = version / 100 - major * 100;
        int patch = version % 100;

        return major + "." + minor + '.' + patch;
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return JnaUtils.getFeatures().contains(capability);
    }

    /** {@inheritDoc} */
    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        return new MxSymbolBlock(manager);
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new MxModel(name, device);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return MxNDManager.getSystemManager().newSubManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return MxNDManager.getSystemManager().newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        return new MxGradientCollector();
    }

    /** {@inheritDoc} */
    @Override
    public ParameterServer newParameterServer(Optimizer optimizer) {
        return Boolean.getBoolean("ai.djl.use_local_parameter_server")
                ? new LocalParameterServer(optimizer)
                : new MxParameterServer(optimizer);
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {
        super.setRandomSeed(seed);
        JnaUtils.randomSeed(seed);
        RandomUtils.RANDOM.setSeed(seed);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append(getEngineName()).append(':').append(getVersion()).append(", capabilities: [\n");
        for (String feature : JnaUtils.getFeatures()) {
            sb.append("\t").append(feature).append(",\n"); // NOPMD
        }
        sb.append("]\nMXNet Library: ").append(LibUtils.getLibName());
        return sb.toString();
    }
}
