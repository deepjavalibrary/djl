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
package ai.djl.tensorflow.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.engine.StandardCapabilities;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SymbolBlock;
import ai.djl.tensorflow.engine.javacpp.JavacppUtils;
import ai.djl.tensorflow.engine.javacpp.LibUtils;
import ai.djl.training.GradientCollector;
import ai.djl.util.RandomUtils;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import org.bytedeco.javacpp.PointerScope;
import org.tensorflow.TensorFlow;
import org.tensorflow.internal.c_api.TFE_Context;
import org.tensorflow.internal.c_api.TF_DeviceList;
import org.tensorflow.internal.c_api.TF_Status;
import org.tensorflow.internal.c_api.global.tensorflow;

/**
 * The {@code TfEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://www.tensorflow.org/">Tensorflow Deep Learning Framework</a>.
 *
 * <p>To get an instance of the {@code TfEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "TensorFlow".
 */
public final class TfEngine extends Engine implements AutoCloseable {

    public static final String ENGINE_NAME = "TensorFlow";
    static final int RANK = 3;

    private static AtomicReference<TFE_Context> eagerSessionHandle;

    private TfEngine() {}

    static TfEngine newInstance() {
        try {
            LibUtils.loadLibrary();
            eagerSessionHandle =
                    new AtomicReference<>(
                            JavacppUtils.createEagerSession(
                                    true, 2, JavacppUtils.getSessionConfig()));
            // call a function from tensorflow-java package to
            // load the native library right here
            // if it throws exception, we can catch it here
            TensorFlow.version();
            return new TfEngine();
        } catch (Throwable t) {
            throw new EngineException("Failed to load TensorFlow native library", t);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new TfModel(name, device);
    }

    /** {@inheritDoc} */
    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        throw new UnsupportedOperationException("TensorFlow does not support empty SymbolBlock");
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
        return TensorFlow.version();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"unchecked", "try"})
    public boolean hasCapability(String capability) {
        if (StandardCapabilities.MKL.equals(capability)) {
            return true;
        } else if (StandardCapabilities.CUDA.equals(capability)) {
            TF_DeviceList deviceList = null;
            try (PointerScope ignore = new PointerScope()) {
                TF_Status status = tensorflow.TF_NewStatus();
                deviceList = tensorflow.TFE_ContextListDevices(eagerSessionHandle.get(), status);
                int deviceCount = tensorflow.TF_DeviceListCount(deviceList);
                for (int i = 0; i < deviceCount; i++) {
                    if (tensorflow.TF_DeviceListName(deviceList, i, status)
                            .getString()
                            .toLowerCase()
                            .contains("gpu")) {
                        return true;
                    }
                }
            } finally {
                // deviceList isn't registered with deallocator
                // so it's not closed by PointerScope
                // close it manually
                synchronized (Objects.requireNonNull(deviceList)) {
                    if (!deviceList.isNull()) {
                        tensorflow.TF_DeleteDeviceList(deviceList);
                    }
                }
            }
            return false;
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return TfNDManager.getSystemManager().newSubManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return TfNDManager.getSystemManager().newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        throw new UnsupportedOperationException("TensorFlow does not support training yet");
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {
        super.setRandomSeed(seed);
        RandomUtils.RANDOM.setSeed(seed);
    }

    TFE_Context getEagerSession() {
        return eagerSessionHandle.get();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append(getEngineName())
                .append(':')
                .append(getVersion())
                .append(", capabilities: [\n\t" + StandardCapabilities.MKL + ",\n");
        if (hasCapability(StandardCapabilities.CUDA)) {
            sb.append("\t").append(StandardCapabilities.CUDA).append(",\n"); // NOPMD
        }
        sb.append("]\nTensorFlow Library: ").append(LibUtils.getLibName());
        return sb.toString();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        TFE_Context handle = eagerSessionHandle.getAndSet(null);
        if (handle != null && !handle.isNull()) {
            handle.close();
        }
    }
}
