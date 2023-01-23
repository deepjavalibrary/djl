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
package ai.djl.fasttext;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.fasttext.jni.FtWrapper;
import ai.djl.fasttext.zoo.nlp.textclassification.FtTextClassification;
import ai.djl.fasttext.zoo.nlp.word_embedding.FtWordEmbeddingBlock;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import ai.djl.util.PairList;
import ai.djl.util.Utils;
import ai.djl.util.passthrough.PassthroughNDManager;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * {@code FtModel} is the fastText implementation of {@link Model}.
 *
 * <p>FtModel contains all the methods in Model to load and process a model. However, it only
 * supports training by using {@link TrainFastText}.
 */
public class FtModel implements Model {

    FtAbstractBlock block;

    private Path modelDir;
    private String modelName;
    private Map<String, String> properties;

    /**
     * Constructs a new Model.
     *
     * @param name the model name
     */
    public FtModel(String name) {
        this.modelName = name;
        properties = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws IOException, MalformedModelException {
        if (Files.notExists(modelPath)) {
            throw new FileNotFoundException(
                    "Model directory doesn't exist: " + modelPath.toAbsolutePath());
        }
        modelDir = modelPath.toAbsolutePath();
        Path modelFile = findModelFile(prefix);
        if (modelFile == null) {
            modelFile = findModelFile(modelDir.toFile().getName());
            if (modelFile == null) {
                throw new FileNotFoundException("No .ftz or .bin file found in : " + modelPath);
            }
        }

        String modelFilePath = modelFile.toString();
        FtWrapper fta = FtWrapper.newInstance();
        if (!fta.checkModel(modelFilePath)) {
            throw new MalformedModelException("Malformed FastText model file:" + modelFilePath);
        }
        fta.loadModel(modelFilePath);

        if (options != null) {
            for (Map.Entry<String, ?> entry : options.entrySet()) {
                properties.put(entry.getKey(), entry.getValue().toString());
            }
        }
        String modelType = fta.getModelType();
        properties.put("model-type", modelType);

        if ("sup".equals(modelType)) {
            String labelPrefix =
                    properties.getOrDefault(
                            "label-prefix", FtTextClassification.DEFAULT_LABEL_PREFIX);
            block = new FtTextClassification(fta, labelPrefix);
            modelDir = block.getModelFile();
        } else if ("cbow".equals(modelType) || "sg".equals(modelType)) {
            block = new FtWordEmbeddingBlock(fta);
            modelDir = block.getModelFile();
        } else {
            throw new MalformedModelException("Unexpected FastText model type: " + modelType);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void load(InputStream modelStream, Map<String, ?> options) {
        throw new UnsupportedOperationException("Not supported.");
    }

    private Path findModelFile(String prefix) {
        if (Files.isRegularFile(modelDir)) {
            Path file = modelDir;
            modelDir = modelDir.getParent();
            String fileName = file.toFile().getName();
            if (fileName.endsWith(".ftz") || fileName.endsWith(".bin")) {
                modelName = fileName.substring(0, fileName.length() - 4);
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
            if (prefix.endsWith(".ftz") || prefix.endsWith(".bin")) {
                return null;
            }
            modelFile = modelDir.resolve(prefix + ".ftz");
            if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                modelFile = modelDir.resolve(prefix + ".bin");
                if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                    return null;
                }
            }
        }
        return modelFile;
    }

    /** {@inheritDoc} */
    @Override
    public void save(Path modelDir, String newModelName) {}

    /** {@inheritDoc} */
    @Override
    public Path getModelPath() {
        return modelDir;
    }

    /** {@inheritDoc} */
    @Override
    public FtAbstractBlock getBlock() {
        return block;
    }

    /** {@inheritDoc} */
    @Override
    public void setBlock(Block block) {
        if (!(block instanceof FtAbstractBlock)) {
            throw new IllegalArgumentException("Expected a FtAbstractBlock Block");
        }
        this.block = (FtAbstractBlock) block;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return modelName;
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        throw new UnsupportedOperationException(
                "FastText only supports training using the FtAbstractBlocks");
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator, Device device) {
        return new Predictor<>(this, translator, device, false);
    }

    /** {@inheritDoc} */
    @Override
    public void setDataType(DataType dataType) {}

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return DataType.UNKNOWN;
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeOutput() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String[] getArtifactNames() {
        return Utils.EMPTY_ARRAY;
    }

    /** {@inheritDoc} */
    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public URL getArtifact(String artifactName) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getArtifactAsStream(String name) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getNDManager() {
        return PassthroughNDManager.INSTANCE;
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
        block.close();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append("Model (\n\tName: ").append(modelName);
        if (modelDir != null) {
            sb.append("\n\tModel location: ").append(modelDir.toAbsolutePath());
        }
        for (Map.Entry<String, String> entry : properties.entrySet()) {
            sb.append("\n\t").append(entry.getKey()).append(": ").append(entry.getValue());
        }
        sb.append("\n)");
        return sb.toString();
    }
}
