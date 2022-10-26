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
import ai.djl.util.Utils;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
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
        if (block != null) {
            throw new UnsupportedOperationException("ONNX Runtime does not support dynamic blocks");
        }
        Path modelFile = findModelFile(prefix);
        if (modelFile == null) {
            modelFile = findModelFile(modelDir.toFile().getName());
            if (modelFile == null) {
                throw new FileNotFoundException(".onnx file not found in: " + modelPath);
            }
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

    private Path findModelFile(String prefix) {
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
        if (prefix == null) {
            prefix = modelName;
        }
        Path modelFile = modelDir.resolve(prefix);
        if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
            if (prefix.endsWith(".onnx")) {
                return null;
            }
            modelFile = modelDir.resolve(prefix + ".onnx");
            if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                return null;
            }
        }
        return modelFile;
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
        if (interOpNumThreads != null) {
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
        if (customOpLibrary != null) {
            ortSession.registerCustomOpLibrary(customOpLibrary);
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
}
