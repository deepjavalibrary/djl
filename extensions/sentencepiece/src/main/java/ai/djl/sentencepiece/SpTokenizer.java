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
package ai.djl.sentencepiece;

import ai.djl.modality.nlp.preprocess.Tokenizer;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * {@code SpTokenizer} is a SentencePiece implementation of the {@link Tokenizer} interface that
 * converts sentences into token.
 */
public class SpTokenizer implements Tokenizer, AutoCloseable {

    private SpProcessor processor;

    /**
     * Create a SentencePiece Tokenizer from existing models.
     *
     * @param modelPath the directory or file path of the model location
     * @throws IOException when IO operation fails in loading a resource
     */
    public SpTokenizer(Path modelPath) throws IOException {
        this(modelPath, null);
    }

    /**
     * Create a SentencePiece Tokenizer from existing models.
     *
     * @param modelPath the directory or file path of the model location
     * @param prefix the model file name or path prefix
     * @throws IOException when IO operation fails in loading a resource
     */
    public SpTokenizer(Path modelPath, String prefix) throws IOException {
        this.processor = SpProcessor.newInstance();
        loadModel(modelPath, prefix);
    }

    /** {@inheritDoc} */
    @Override
    public List<String> tokenize(String sentence) {
        return Arrays.asList(processor.tokenize(sentence));
    }

    /** {@inheritDoc} */
    @Override
    public String buildSentence(List<String> tokens) {
        return processor.buildSentence(tokens.toArray(new String[0]));
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        processor.close();
    }

    /**
     * Get SentencePiece processor.
     *
     * @return {@link SpProcessor}
     */
    public SpProcessor getProcessor() {
        return processor;
    }

    private void loadModel(Path modelPath, String prefix) throws IOException {
        if (Files.notExists(modelPath)) {
            throw new FileNotFoundException(
                    "Model path doesn't exist: " + modelPath.toAbsolutePath());
        }
        Path modelDir = modelPath.toAbsolutePath();
        Path modelFile = findModelFile(modelDir, prefix);
        if (modelFile == null) {
            // TODO: support proto and IOStream model
            modelFile = findModelFile(modelDir, modelDir.toFile().getName());
            if (modelFile == null) {
                throw new FileNotFoundException("No .model found in : " + modelPath);
            }
        }

        String modelFilePath = modelFile.toString();
        processor.loadModel(modelFilePath);
    }

    private Path findModelFile(Path modelPath, String prefix) {
        if (Files.isRegularFile(modelPath)) {
            return modelPath;
        }
        Path modelFile = modelPath.resolve(prefix);
        if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
            if (prefix.endsWith(".model")) {
                return null;
            }
            modelFile = modelPath.resolve(prefix + ".model");
            if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                return null;
            }
        }
        return modelFile;
    }
}
