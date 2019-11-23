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
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import ai.djl.util.Utils;
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
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@code MxModel} is the MXNet implementation of {@link Model}.
 *
 * <p>MxModel contains all the methods in Model to load and process a model. In addition, it
 * provides MXNet Specific functionality, such as getSymbol to obtain the Symbolic graph and
 * getParameters to obtain the parameter NDArrays
 */
public class MxModel implements Model {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    private static final int MODEL_VERSION = 1;

    private Path modelDir;
    private String modelName;
    private MxNDManager manager;
    private Block block;
    private DataType dataType;
    private Map<String, String> properties;
    private PairList<String, Shape> inputData;
    private Map<String, Object> artifacts = new ConcurrentHashMap<>();
    // the variable is used to avoid ParameterStore copy for the first time
    private AtomicBoolean first;

    /**
     * Constructs a new Model on a given device.
     *
     * @param device the device the model should be located on
     */
    MxModel(Device device) {
        device = Device.defaultIfNull(device);
        dataType = DataType.FLOAT32;
        properties = new ConcurrentHashMap<>();
        manager = MxNDManager.getSystemManager().newSubManager(device);
        first = new AtomicBoolean(true);
    }

    /**
     * Loads the MXNet model from a specified location.
     *
     * <p>MXNet engine looks for {MODEL_NAME}-symbol.json and {MODEL_NAME}-{EPOCH}.params files in
     * the specified directory. By default, MXNet engine will pick up the latest epoch of the
     * parameter file. However, users can explicitly specify an epoch to be loaded:
     *
     * <pre>
     * Map&lt;String, String&gt; options = new HashMap&lt;&gt;()
     * <b>options.put("epoch", "3");</b>
     * model.load(modelPath, "squeezenet", options);
     * </pre>
     *
     * @param modelPath the directory of the model
     * @param modelName the name/prefix of the model
     * @param options load model options, see documentation for the specific engine
     * @throws IOException Exception for file loading
     */
    @Override
    public void load(Path modelPath, String modelName, Map<String, String> options)
            throws IOException, MalformedModelException {
        modelDir = modelPath.toAbsolutePath();
        this.modelName = modelName;
        if (block == null) {
            // load MxSymbolBlock
            Path symbolFile = modelDir.resolve(modelName + "-symbol.json");
            if (Files.notExists(symbolFile)) {
                throw new FileNotFoundException(
                        "Symbol file not found in: " + modelPath + ", please set block manually.");
            }
            Symbol symbol = Symbol.load(manager, symbolFile.toAbsolutePath().toString());
            // TODO: change default name "data" to model-specific one
            block = new MxSymbolBlock(manager, symbol);
        }
        loadParameters(modelName, options);
        // TODO: Check if Symbol has all names that params file have
    }

    /** {@inheritDoc} */
    @Override
    public void save(Path modelPath, String modelName) throws IOException {
        if (Files.notExists(modelPath)) {
            Files.createDirectories(modelPath);
        }

        if (block == null || !block.isInitialized()) {
            throw new IllegalStateException("Model has not be trained or loaded yet.");
        }

        String epochValue = getProperty("Epoch");
        int epoch =
                epochValue == null
                        ? Utils.getCurrentEpoch(modelPath, modelName) + 1
                        : Integer.parseInt(epochValue);

        Path paramFile = modelPath.resolve(String.format("%s-%04d.params", modelName, epoch));
        try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(paramFile))) {
            dos.writeBytes("DJL@");
            dos.writeInt(MODEL_VERSION);
            dos.writeUTF(modelName);
            dos.writeUTF(dataType.name());
            inputData = block.describeInput();
            dos.writeInt(inputData.size());
            for (Pair<String, Shape> desc : inputData) {
                String name = desc.getKey();
                if (name == null) {
                    dos.writeUTF("");
                } else {
                    dos.writeUTF(name);
                }
                dos.write(desc.getValue().getEncoded());
            }

            dos.writeInt(properties.size());
            for (Map.Entry<String, String> entry : properties.entrySet()) {
                dos.writeUTF(entry.getKey());
                dos.writeUTF(entry.getValue());
            }

            block.saveParameters(dos);
        }
        this.modelName = modelName;
        modelDir = modelPath.toAbsolutePath();
    }

    /** {@inheritDoc} */
    @Override
    public Block getBlock() {
        return block;
    }

    /** {@inheritDoc} */
    @Override
    public void setBlock(Block block) {
        this.block = block;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return modelName;
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        Initializer initializer = trainingConfig.getInitializer();
        block.setInitializer(initializer);

        return new MxTrainer(this, trainingConfig);
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        boolean firstPredictor = first.getAndSet(false);
        boolean shouldCopyParameters = !JnaUtils.useThreadSafePredictor() && !firstPredictor;
        return new MxPredictor<>(this, translator, shouldCopyParameters);
    }

    /** {@inheritDoc} */
    @Override
    public void setDataType(DataType dataType) {
        this.dataType = dataType;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        if (inputData == null) {
            inputData = block.describeInput();
        }
        return inputData;
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeOutput() {
        List<String> names = inputData.keys();
        Shape[] outputShapes =
                block.getOutputShapes(
                        manager, inputData.values().toArray(new Shape[inputData.size()]));
        return new PairList<>(names, Arrays.asList(outputShapes));
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
        throw new FileNotFoundException("File not found: " + file);
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getArtifactAsStream(String name) throws IOException {
        URL url = getArtifact(name);
        return url.openStream();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getNDManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public void setProperty(String key, String value) {
        properties.put(key, value);
    }

    /** {@inheritDoc} */
    @Override
    public String getProperty(String key) {
        return properties.get(key);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        // TODO workaround for MXNet Engine crash issue
        JnaUtils.waitAll();
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

    @SuppressWarnings("PMD.UseConcurrentHashMap")
    private void loadParameters(String modelName, Map<String, String> options)
            throws IOException, MalformedModelException {
        Path paramFile;
        if (Files.isRegularFile(modelDir)) {
            paramFile = modelDir;
        } else {
            String epochOption = null;
            if (options != null) {
                epochOption = options.get("epoch");
            }
            int epoch;
            if (epochOption == null) {
                epoch = Utils.getCurrentEpoch(modelDir, modelName);
                if (epoch == -1) {
                    throw new IOException(
                            "Parameter file not found in: "
                                    + modelDir
                                    + ". If you only specified model path, make sure path name match"
                                    + "your saved model file name.");
                }
            } else {
                epoch = Integer.parseInt(epochOption);
            }

            paramFile = modelDir.resolve(String.format("%s-%04d.params", modelName, epoch));
        }
        logger.debug("Try to load model from {}", paramFile);
        if (readParameters(paramFile)) {
            return;
        }
        logger.debug("DJL formatted model not found, try to find MXNet model");
        NDList paramNDlist =
                JnaUtils.loadNdArray(manager, paramFile.toAbsolutePath(), manager.getDevice());

        MxSymbolBlock symbolBlock = (MxSymbolBlock) block;

        List<Parameter> parameters = symbolBlock.getAllParameters();
        Map<String, Parameter> map = new LinkedHashMap<>();
        parameters.forEach(p -> map.put(p.getName(), p));

        for (NDArray nd : paramNDlist) {
            String key = nd.getName();
            if (key == null) {
                throw new IllegalArgumentException("Array names must be present in parameter file");
            }

            String paramName = key.split(":", 2)[1];
            Parameter parameter = map.remove(paramName);
            parameter.setArray(nd);
        }
        symbolBlock.setInputNames(new ArrayList<>(map.keySet()));

        // TODO: Find a better to infer model DataType from SymbolBlock.
        dataType = paramNDlist.head().getDataType();
        logger.debug("MXNet Model {} ({}) loaded successfully.", modelName, dataType);
    }

    private boolean readParameters(Path paramFile) throws IOException, MalformedModelException {
        try (DataInputStream dis = new DataInputStream(Files.newInputStream(paramFile))) {
            byte[] buf = new byte[4];
            dis.readFully(buf);
            if (!"DJL@".equals(new String(buf, StandardCharsets.US_ASCII))) {
                return false;
            }

            int version = dis.readInt();
            if (version != MODEL_VERSION) {
                throw new IOException("Unsupported model version: " + version);
            }

            modelName = dis.readUTF();
            logger.debug("Loading model parameter: {}", modelName);

            dataType = DataType.valueOf(dis.readUTF());

            int numberOfInputs = dis.readInt();
            inputData = new PairList<>();
            for (int i = 0; i < numberOfInputs; ++i) {
                String inputName = dis.readUTF(); // input name
                Shape shape = Shape.decode(dis);
                inputData.add(inputName, shape);
            }

            int numberOfProperties = dis.readInt();
            for (int i = 0; i < numberOfProperties; ++i) {
                String key = dis.readUTF();
                String value = dis.readUTF();
                properties.put(key, value);
            }

            block.loadParameters(manager, dis);
            logger.debug("DJL model loaded successfully");
        }
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append("Model (\n\tName: ").append(modelName);
        if (modelDir != null) {
            sb.append("\n\tModel location: ").append(modelDir.toAbsolutePath());
        }
        sb.append("\n\tData Type: ").append(dataType);
        for (Map.Entry<String, String> entry : properties.entrySet()) {
            sb.append("\n\t").append(entry.getKey()).append(": ").append(entry.getValue());
        }
        sb.append("\n)");
        return sb.toString();
    }
}
