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
package ai.djl.onnxruntime.engine;

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Utils;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * {@code OrtModel} is the ONNX Runtime implementation of {@link Model}.
 *
 * <p>OrtModel contains all the methods in Model to load and process a model. In addition, it
 * provides ONNX Runtime Specific functionality
 */
public class OrtModel extends BaseModel {

    private static final Logger logger = LoggerFactory.getLogger(OrtModel.class);

    private OrtEnvironment env;
    private SessionOptions sessionOptions;

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     * @param manager the {@link NDManager} to holds the NDArray
     * @param env the {@link OrtEnvironment} ONNX Environment to create session
     */
    OrtModel(String name, NDManager manager, OrtEnvironment env) {
        super(name);
        this.manager = manager;
        this.manager.setName("ortModel");
        this.env = env;
        dataType = DataType.FLOAT32;
        sessionOptions = new SessionOptions();
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws IOException, MalformedModelException {
        setModelDir(modelPath);
        wasLoaded = true;
        if (block != null) {
            throw new UnsupportedOperationException("ONNX Runtime does not support dynamic blocks");
        }

        Path modelFile;
        if (prefix != null) {
            modelFile = findModelFile(prefix);
        } else {
            // search for .onnx file with folder name or "model.onnx"
            modelFile = findModelFile(modelName, modelDir.toFile().getName(), "model.onnx");
        }

        if (modelFile == null) {
            throw new FileNotFoundException(".onnx file not found in: " + modelPath);
        }

        try {
            SessionOptions ortOptions = getSessionOptions(options);
            OrtSession session = env.createSession(modelFile.toString(), ortOptions);
            block = new OrtSymbolBlock(session, (OrtNDManager) manager);
        } catch (OrtException e) {
            throw new MalformedModelException("ONNX Model cannot be loaded", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void load(InputStream is, Map<String, ?> options)
            throws IOException, MalformedModelException {
        if (block != null) {
            throw new UnsupportedOperationException("ONNX Runtime does not support dynamic blocks");
        }
        modelDir = Files.createTempDirectory("ort-model");
        modelDir.toFile().deleteOnExit();
        try {
            byte[] buf = Utils.toByteArray(is);
            SessionOptions ortOptions = getSessionOptions(options);
            OrtSession session = env.createSession(buf, ortOptions);
            block = new OrtSymbolBlock(session, (OrtNDManager) manager);
        } catch (OrtException e) {
            throw new MalformedModelException("ONNX Model cannot be loaded", e);
        }
    }

    private Path findModelFile(String... prefixes) {
        if (Files.isRegularFile(modelDir)) {
            Path file = modelDir;
            modelDir = modelDir.getParent();
            String fileName = file.toFile().getName();
            if (fileName.endsWith(".onnx")) {
                modelName = fileName.substring(0, fileName.length() - 5);
            } else {
                modelName = fileName;
            }
            return file;
        }
        for (String prefix : prefixes) {
            Path modelFile = modelDir.resolve(prefix);
            if (Files.isRegularFile(modelFile)) {
                return modelFile;
            }
            if (!prefix.endsWith(".onnx")) {
                modelFile = modelDir.resolve(prefix + ".onnx");
                if (Files.isRegularFile(modelFile)) {
                    return modelFile;
                }
            }
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        try {
            sessionOptions.close();
        } catch (IllegalArgumentException ignore) {
            // ignore
        }
    }

    private SessionOptions getSessionOptions(Map<String, ?> options) throws OrtException {
        if (options == null) {
            return sessionOptions;
        }

        SessionOptions ortSession = sessionOptions;
        if (options.containsKey("sessionOptions")) {
            ortSession = (SessionOptions) options.get("sessionOptions");
        }

        String interOpNumThreads = (String) options.get("interOpNumThreads");
        if (interOpNumThreads != null) {
            ortSession.setInterOpNumThreads(Integer.parseInt(interOpNumThreads));
        }
        String intraOpNumThreads = (String) options.get("intraOpNumThreads");
        if (intraOpNumThreads != null) {
            ortSession.setIntraOpNumThreads(Integer.parseInt(intraOpNumThreads));
        }
        String executionMode = (String) options.get("executionMode");
        if (executionMode != null) {
            ortSession.setExecutionMode(ExecutionMode.valueOf(executionMode));
        }
        String optLevel = (String) options.get("optLevel");
        if (optLevel != null) {
            ortSession.setOptimizationLevel(OptLevel.valueOf(optLevel));
        }
        String memoryOptimization = (String) options.get("memoryPatternOptimization");
        if (Boolean.parseBoolean(memoryOptimization)) {
            ortSession.setMemoryPatternOptimization(true);
        }

        String cpuArena = (String) options.get("cpuArenaAllocator");
        if (Boolean.parseBoolean(cpuArena)) {
            ortSession.setCPUArenaAllocator(true);
        }

        String disablePerSessionThreads = (String) options.get("disablePerSessionThreads");
        if (Boolean.parseBoolean(disablePerSessionThreads)) {
            ortSession.disablePerSessionThreads();
        }

        String customOpLibrary = (String) options.get("customOpLibrary");
        if (customOpLibrary == null) {
            customOpLibrary = getOrtxLibraryPath();
        }
        if (customOpLibrary != null) {
            ortSession.registerCustomOpLibrary(customOpLibrary);
        }

        String profilerOutput = (String) options.get("profilerOutput");
        if (profilerOutput != null) {
            ortSession.enableProfiling(profilerOutput);
        }

        Device device = manager.getDevice();
        if (options.containsKey("ortDevice")) {
            String ortDevice = (String) options.get("ortDevice");
            switch (ortDevice) {
                case "TensorRT":
                    if (!device.isGpu()) {
                        throw new IllegalArgumentException("TensorRT required GPU device.");
                    }
                    ortSession.addTensorrt(device.getDeviceId());
                    break;
                case "ROCM":
                    ortSession.addROCM();
                    break;
                case "CoreML":
                    ortSession.addCoreML();
                    break;
                default:
                    throw new IllegalArgumentException("Invalid ortDevice: " + ortDevice);
            }
        } else if (device.isGpu()) {
            ortSession.addCUDA(device.getDeviceId());
        }
        return ortSession;
    }

    private String getOrtxLibraryPath() {
        ClassLoader cl = ClassLoaderUtils.getContextClassLoader();
        try {
            Class<?> clazz = Class.forName("ai.onnxruntime.extensions.OrtxPackage", true, cl);
            Method method = clazz.getDeclaredMethod("getLibraryPath");
            return (String) method.invoke(null);
        } catch (Throwable e) {
            logger.info("Onnx extension not found in classpath.");
            logger.trace("Failed to load onnx extension", e);
        }
        return null;
    }
}
