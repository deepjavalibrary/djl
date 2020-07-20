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

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.training.TrainingResult;
import com.github.jfasttext.FastTextWrapper;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.bytedeco.javacpp.CharPointer;
import org.bytedeco.javacpp.PointerPointer;

/**
 * {@code FtModel} is the fastText implementation of {@link Model}.
 *
 * <p>FtModel contains all the methods in Model to load and process a model.
 */
public class FtModel implements AutoCloseable {

    FastTextWrapper.FastTextApi fta;

    private Path modelDir;
    private String modelName;

    /**
     * Constructs a new Model.
     *
     * @param name the model name
     */
    public FtModel(String name) {
        this.modelName = name;
        fta = new FastTextWrapper.FastTextApi();
    }

    /**
     * Loads the fastText model from a specified location.
     *
     * @param modelPath the directory of the model
     * @param prefix the model file name or path prefix
     * @throws IOException Exception for file loading
     * @throws MalformedModelException if model file is corrupted
     */
    public void load(Path modelPath, String prefix) throws IOException, MalformedModelException {
        if (Files.notExists(modelPath)) {
            throw new FileNotFoundException(
                    "Model directory doesn't exist: " + modelPath.toAbsolutePath());
        }
        modelDir = modelPath.toAbsolutePath();
        if (prefix == null) {
            prefix = modelName;
        }
        Path modelFile = findModelFile(prefix);
        if (modelFile == null) {
            modelFile = findModelFile(modelDir.toFile().getName());
            if (modelFile == null) {
                throw new FileNotFoundException("No .ftz or .bin file found in : " + modelPath);
            }
        }

        String modelFilePath = modelFile.toString();
        if (!fta.checkModel(modelFilePath)) {
            throw new MalformedModelException("Malformed FastText model file:" + modelFilePath);
        }
        fta.loadModel(modelFilePath);
    }

    private Path findModelFile(String prefix) {
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

    /**
     * Train the fastText model.
     *
     * @param config the training configuration to use
     * @param inputFilePath the training dataset file path
     * @return the result of the training
     * @throws IOException when IO operation fails in loading a resource
     */
    public TrainingResult fit(FtTrainingConfig config, Path inputFilePath) throws IOException {
        Path outputDir = config.getOutputDir();
        if (Files.notExists(outputDir)) {
            Files.createDirectory(outputDir);
        }
        String fitModelName = config.getModelName();
        Path modelFile = outputDir.resolve(fitModelName).toAbsolutePath();

        String[] args = config.toCommand(inputFilePath.toString());

        fta.runCmd(args.length, new PointerPointer<CharPointer>(args));
        setModelFile(modelFile);

        TrainingResult result = new TrainingResult();
        int epoch = config.getEpoch();
        if (epoch <= 0) {
            epoch = 5;
        }
        result.setEpoch(epoch);
        return result;
    }

    private void setModelFile(Path modelFile) {
        this.modelDir = modelFile;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        fta.unloadModel();
        fta.close();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append("Model (\n\tName: ").append(modelName);
        if (modelDir != null) {
            sb.append("\n\tModel location: ").append(modelDir.toAbsolutePath());
        }
        sb.append("\n)");
        return sb.toString();
    }
}
