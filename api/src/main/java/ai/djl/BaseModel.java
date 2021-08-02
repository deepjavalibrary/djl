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
package ai.djl;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import ai.djl.util.Utils;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
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
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** {@code BaseModel} is the basic implementation of {@link Model}. */
public abstract class BaseModel implements Model {

    private static final Logger logger = LoggerFactory.getLogger(BaseModel.class);
    private static final int MODEL_VERSION = 1;

    protected Path modelDir;
    protected Block block;
    protected String modelName;
    protected NDManager manager;
    protected DataType dataType;
    protected PairList<String, Shape> inputData;
    protected Map<String, Object> artifacts = new ConcurrentHashMap<>();
    protected Map<String, String> properties = new ConcurrentHashMap<>();

    protected BaseModel(String modelName) {
        this.modelName = modelName;
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
    public NDManager getNDManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        return new Predictor<>(this, translator, false);
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
    public void close() {
        manager.close();
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
        if (block instanceof SymbolBlock) {
            return ((SymbolBlock) block).describeOutput();
        }
        // create fake input to calculate output shapes
        NDList input = new NDList();
        for (Pair<String, Shape> pair : describeInput()) {
            input.add(manager.ones(pair.getValue()));
        }
        List<String> outputNames = new ArrayList<>();
        NDList output = block.forward(new ParameterStore(manager, true), input, false);
        Shape[] outputShapes = output.stream().map(NDArray::getShape).toArray(Shape[]::new);
        for (int i = 0; i < outputShapes.length; i++) {
            outputNames.add("output" + i);
        }
        return new PairList<>(outputNames, Arrays.asList(outputShapes));
    }

    /** {@inheritDoc} */
    @Override
    public String[] getArtifactNames() {
        throw new UnsupportedOperationException("Not supported!");
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
        return new BufferedInputStream(url.openStream());
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

    protected void setModelDir(Path modelDir) {
        this.modelDir = modelDir.toAbsolutePath();
    }

    /** {@inheritDoc} */
    @Override
    public void save(Path modelPath, String newModelName) throws IOException {
        if (newModelName == null || newModelName.isEmpty()) {
            newModelName = modelName;
        }
        if (Files.notExists(modelPath)) {
            Files.createDirectories(modelPath);
        }

        if (block == null || !block.isInitialized()) {
            throw new IllegalStateException("Model has not be trained or loaded yet.");
        }

        String epochValue = getProperty("Epoch");
        int epoch =
                epochValue == null
                        ? Utils.getCurrentEpoch(modelPath, newModelName) + 1
                        : Integer.parseInt(epochValue);

        String fileName = String.format(Locale.ROOT, "%s-%04d.params", newModelName, epoch);
        Path paramFile = modelPath.resolve(fileName);
        try (DataOutputStream dos =
                new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(paramFile)))) {
            dos.writeBytes("DJL@");
            dos.writeInt(MODEL_VERSION);
            dos.writeUTF(newModelName);
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
        modelDir = modelPath.toAbsolutePath();
    }

    /** {@inheritDoc} */
    @Override
    public Path getModelPath() {
        return modelDir;
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

    /** {@inheritDoc} */
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        if (manager.isOpen()) {
            logger.warn("Model: {} was not closed explicitly.", modelName);
            manager.close();
        }
        super.finalize();
    }

    protected Path paramPathResolver(String prefix, Map<String, ?> options) throws IOException {
        Object epochOption = null;
        if (options != null) {
            epochOption = options.get("epoch");
        }
        int epoch;
        if (epochOption == null) {
            epoch = Utils.getCurrentEpoch(modelDir, prefix);
            if (epoch == -1) {
                return null;
            }
        } else {
            epoch = Integer.parseInt(epochOption.toString());
        }

        return modelDir.resolve(String.format(Locale.ROOT, "%s-%04d.params", prefix, epoch));
    }

    protected boolean readParameters(Path paramFile, Map<String, ?> options)
            throws IOException, MalformedModelException {
        logger.debug("Try to load model from {}", paramFile);
        try (DataInputStream dis =
                new DataInputStream(new BufferedInputStream(Files.newInputStream(paramFile)))) {
            byte[] buf = new byte[4];
            dis.readFully(buf);
            if (!"DJL@".equals(new String(buf, StandardCharsets.US_ASCII))) {
                return false;
            }

            int version = dis.readInt();
            if (version != MODEL_VERSION) {
                throw new IOException("Unsupported model version: " + version);
            }

            String savedModelName = dis.readUTF();
            logger.debug("Loading saved model: {} parameter", savedModelName);

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
}
