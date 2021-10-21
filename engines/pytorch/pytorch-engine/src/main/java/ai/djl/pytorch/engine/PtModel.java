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
package ai.djl.pytorch.engine;

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * {@code PtModel} is the PyTorch implementation of {@link Model}.
 *
 * <p>PtModel contains all the methods in Model to load and process a model. In addition, it
 * provides PyTorch Specific functionality
 */
public class PtModel extends BaseModel {

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     * @param device the device the model should be located on
     */
    PtModel(String name, Device device) {
        super(name);
        manager = PtNDManager.getSystemManager().newSubManager(device);
        manager.setName("ptModel");
        dataType = DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws IOException, MalformedModelException {
        modelDir = modelPath.toAbsolutePath();
        if (prefix == null) {
            prefix = modelName;
        }

        if (block == null) {
            Path modelFile = findModelFile(prefix);
            if (modelFile == null) {
                modelFile = findModelFile(modelDir.toFile().getName());
                if (modelFile == null) {
                    String fileName = prefix.endsWith(".pt") ? prefix : prefix + ".pt";
                    throw new FileNotFoundException(fileName + " file not found in: " + modelDir);
                }
            }
            String[] extraFileKeys = new String[0];
            String[] extraFileValues = new String[0];
            boolean mapLocation = false;
            // load jit extra files
            if (options != null) {
                if (options.containsKey("extraFiles")) {
                    extraFileKeys = ((String) options.get("extraFiles")).split(",");
                    extraFileValues = new String[extraFileKeys.length];
                }
                mapLocation = Boolean.parseBoolean((String) options.get("mapLocation"));
            }
            block =
                    JniUtils.loadModule(
                            (PtNDManager) manager,
                            modelFile,
                            mapLocation,
                            extraFileKeys,
                            extraFileValues);
            for (int i = 0; i < extraFileKeys.length; i++) {
                properties.put(extraFileKeys[i], extraFileValues[i]);
            }
        } else {
            Path paramFile = paramPathResolver(prefix, options);
            if (paramFile == null) {
                throw new IOException(
                        "Parameter file not found in: "
                                + modelDir
                                + ". If you only specified model path, make sure path name match"
                                + "your saved model file name.");
            }
            readParameters(paramFile, options);
        }
    }

    /**
     * Load PyTorch model from {@link InputStream}.
     *
     * <p>Currently, only TorchScript file are supported
     *
     * @param modelStream the stream of the model file
     * @throws IOException model loading error
     */
    public void load(InputStream modelStream) throws IOException {
        load(modelStream, true);
    }

    /**
     * Load PyTorch model from {@link InputStream}.
     *
     * @param modelStream the stream of the model file
     * @param mapLocation force load to specified device if true
     * @throws IOException model loading error
     */
    public void load(InputStream modelStream, boolean mapLocation) throws IOException {
        block = JniUtils.loadModule((PtNDManager) manager, modelStream, mapLocation, false);
    }

    private Path findModelFile(String prefix) {
        if (Files.isRegularFile(modelDir)) {
            Path file = modelDir;
            modelDir = modelDir.getParent();
            String fileName = file.toFile().getName();
            if (fileName.endsWith(".pt")) {
                modelName = fileName.substring(0, fileName.length() - 3);
            } else {
                modelName = fileName;
            }
            return file;
        }
        Path modelFile = modelDir.resolve(prefix);
        if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
            if (prefix.endsWith(".pt")) {
                return null;
            }
            modelFile = modelDir.resolve(prefix + ".pt");
            if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                return null;
            }
        }
        return modelFile;
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        PairList<Initializer, Predicate<Parameter>> initializer = trainingConfig.getInitializers();
        if (block == null) {
            throw new IllegalStateException(
                    "You must set a block for the model before creating a new trainer");
        }
        for (Pair<Initializer, Predicate<Parameter>> pair : initializer) {
            if (pair.getKey() != null && pair.getValue() != null) {
                block.setInitializer(pair.getKey(), pair.getValue());
            }
        }

        return new Trainer(this, trainingConfig);
    }

    /** {@inheritDoc} */
    @Override
    public String[] getArtifactNames() {
        try {
            List<Path> files =
                    Files.walk(modelDir).filter(Files::isRegularFile).collect(Collectors.toList());
            List<String> ret = new ArrayList<>(files.size());
            for (Path path : files) {
                String fileName = path.toFile().getName();
                if (fileName.endsWith(".pt")) {
                    // ignore model files.
                    continue;
                }
                Path relative = modelDir.relativize(path);
                ret.add(relative.toString());
            }
            return ret.toArray(new String[0]);
        } catch (IOException e) {
            throw new AssertionError("Failed list files", e);
        }
    }
}
