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
package org.apache.mxnet.engine;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.nn.MxSymbolBlock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.Device;
import software.amazon.ai.Model;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.translate.Translator;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.Utils;

/**
 * {@code MxModel} is MXNet implementation of {@link Model}.
 *
 * <p>MxModel contains all methods in Model to load and process model. In addition, it provides
 * MXNet Specific functionality, such as getSymbol to obtain the Symbolic graph and getParameters to
 * obtain the parameter NDArrays
 */
public class MxModel implements Model {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    private static final int MODEL_VERSION = 1;

    private Path modelDir;
    private MxNDManager manager;
    private Block block;
    private DataDesc[] inputData;
    private Map<String, Object> artifacts = new ConcurrentHashMap<>();

    MxModel(Device device) {
        device = Device.defaultIfNull(device);
        manager = MxNDManager.getSystemManager().newSubManager(device);
    }

    /**
     * Load the MXNet model from specified location.
     *
     * <p>MXNet engine looks for modelName.json and modelName-xxxx.params files in specified
     * directory. By default, MXNet engine will pick up latest epoch of parameter file. However,
     * user can explicitly an epoch to be loaded:
     *
     * <pre>
     * Map&lt;String, String&gt; options = new HashMap&lt;&gt;()
     * <b>options.put("epoch", "3");</b>
     * model.load(modelPath, "squeezenet", options);
     * </pre>
     *
     * @param modelPath Directory of the model
     * @param modelName Name/Prefix of the model
     * @param options load model options, check document for specific engine
     * @throws IOException Exception for file loading
     */
    @Override
    public void load(Path modelPath, String modelName, Map<String, String> options)
            throws IOException {
        MxEngine engine = ((MxEngine) Engine.getEngine(MxEngine.ENGINE_NAME));
        engine.setNumpyMode(false);

        try {
            modelDir = modelPath.toAbsolutePath();
            if (block == null) {
                Path symbolFile = modelDir.resolve(modelName + "-symbol.json");
                if (Files.notExists(symbolFile)) {
                    throw new FileNotFoundException(
                            "Symbol file not found in: "
                                    + modelPath
                                    + ", please set block manually.");
                }
                Symbol symbol = Symbol.load(manager, symbolFile.toAbsolutePath().toString());
                block = new MxSymbolBlock(manager, symbol);
            }

            loadParameters(modelName, options);

            // TODO: Check if Symbol has all names that params file have
        } finally {
            engine.setNumpyMode(true);
        }
    }

    @Override
    public void save(Path modelPath, String modelName) throws IOException {
        if (Files.notExists(modelPath)) {
            Files.createDirectories(modelPath);
        }

        if (block == null || !block.isInitialized()) {
            throw new IllegalStateException("Model has not be trained or loaded yet.");
        }

        int epoch = Utils.getCurrentEpoch(modelDir, modelName) + 1;
        Path paramFile = modelDir.resolve(String.format("%s-%04d.params", modelName, epoch));
        try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(paramFile))) {
            dos.writeBytes("JOUL");
            dos.writeInt(MODEL_VERSION);
            dos.writeUTF(modelName);
            DataDesc[] descs = block.describeInput();
            dos.writeInt(descs.length);
            for (DataDesc desc : descs) {
                String name = desc.getName();
                if (name == null) {
                    dos.writeUTF("");
                } else {
                    dos.writeUTF(name);
                }

                dos.writeUTF(desc.getDataType().name());
                dos.write(desc.getShape().getEncoded());
            }

            block.saveParameters(dos);
        }
    }

    @Override
    public Block getBlock() {
        return block;
    }

    @Override
    public void setBlock(Block block) {
        this.block = block;
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        Initializer initializer = trainingConfig.getInitializer();
        Device[] devices = trainingConfig.getDevices();
        boolean overwrite = trainingConfig.isOverwriteInitializer();
        block.setInitializer(manager, initializer, overwrite, devices);

        return new MxTrainer(this, trainingConfig);
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        return new MxPredictor<>(this, translator, manager.getDevice());
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        block.cast(dataType);
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeInput() {
        if (inputData == null) {
            inputData = block.describeInput();
        }
        return inputData;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeOutput() {
        return null;
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
                if (fileName.endsWith(".params") || fileName.endsWith("-symbol.json")) {
                    // ignore symbol and param files.
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

    /** {@inheritDoc} */
    @SuppressWarnings("unchecked")
    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) throws IOException {
        try {
            Object artifact =
                    artifacts.computeIfAbsent(
                            name,
                            v -> {
                                try (InputStream is = getArtifactAsStream(name)) {
                                    return function.apply(is);
                                } catch (IOException e) {
                                    throw new IllegalStateException(e);
                                }
                            });
            return (T) artifact;
        } catch (RuntimeException e) {
            Throwable t = e.getCause();
            if (t instanceof IOException) {
                throw (IOException) e.getCause();
            }
            throw e;
        }
    }

    /** {@inheritDoc} */
    @Override
    public URL getArtifact(String artifactName) throws IOException {
        if (artifactName == null) {
            throw new IllegalArgumentException("artifactName cannot be null");
        }
        Path file = modelDir.resolve(artifactName);
        if (Files.exists(file) && Files.isReadable(file)) {
            return file.toUri().toURL();
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getArtifactAsStream(String name) throws IOException {
        URL url = getArtifact(name);
        if (url == null) {
            return null;
        }
        return url.openStream();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getNDManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        manager.close();
    }

    /** {@inheritDoc} */
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        if (manager.isOpen()) {
            logger.warn("MxModel was not closed explicitly.");
            manager.close();
        }
        super.finalize();
    }

    private void loadParameters(String modelName, Map<String, String> options) throws IOException {
        String epochOption = null;
        if (options != null) {
            epochOption = options.get("epoch");
        }
        int epoch;
        if (epochOption == null) {
            epoch = Utils.getCurrentEpoch(modelDir, modelName);
            if (epoch == -1) {
                throw new IOException("Parameter file not found in: " + modelDir);
            }
        } else {
            epoch = Integer.parseInt(epochOption);
        }

        Path paramFile = modelDir.resolve(String.format("%s-%04d.params", modelName, epoch));
        if (readParameters(paramFile)) {
            return;
        }

        NDList paramNDlist = JnaUtils.loadNdArray(manager, paramFile.toAbsolutePath());
        Device device = manager.getDevice();

        List<Parameter> parameters = block.getDirectParameters();
        for (Pair<String, NDArray> pair : paramNDlist) {
            String key = pair.getKey();
            if (key == null) {
                throw new IllegalArgumentException("Array names must be present in parameter file");
            }
            String paramName = key.split(":", 2)[1];
            NDArray array = pair.getValue().asInDevice(device, false);
            parameters.add(new Parameter(paramName, block, array));
        }

        if (!device.equals(Device.cpu())) {
            // MXNet always load parameters on CPU, we only close parameters if we copied them.
            paramNDlist.close();
        }
    }

    private boolean readParameters(Path paramFile) throws IOException {
        try (DataInputStream dis = new DataInputStream(Files.newInputStream(paramFile))) {
            byte[] buf = new byte[4];
            dis.readFully(buf);
            if (!"JOUL".equals(new String(buf, StandardCharsets.US_ASCII))) {
                return false;
            }

            int version = dis.readInt();
            if (version != MODEL_VERSION) {
                throw new IOException("Unsupported model version: " + version);
            }

            String modelName = dis.readUTF();
            logger.debug("Loading model parameter: {}", modelName);

            int numberOfInputs = dis.readInt();
            for (int i = 0; i < numberOfInputs; ++i) {
                // TODO: store header information in model
                dis.readUTF(); // input name
                dis.readUTF(); // DataType
                Shape.decode(dis);
            }

            block.loadParameters(manager, dis);
        }
        return true;
    }
}
