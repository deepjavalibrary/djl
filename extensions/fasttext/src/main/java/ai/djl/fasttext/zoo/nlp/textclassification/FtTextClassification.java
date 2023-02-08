/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.fasttext.zoo.nlp.textclassification;

import ai.djl.fasttext.FtAbstractBlock;
import ai.djl.fasttext.FtTrainingConfig;
import ai.djl.fasttext.jni.FtWrapper;
import ai.djl.fasttext.zoo.nlp.word_embedding.FtWordEmbeddingBlock;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterStore;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RawDataset;
import ai.djl.util.PairList;
import ai.djl.util.passthrough.PassthroughNDArray;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** A {@link FtAbstractBlock} for {@link ai.djl.Application.NLP#TEXT_CLASSIFICATION}. */
public class FtTextClassification extends FtAbstractBlock {

    public static final String DEFAULT_LABEL_PREFIX = "__label__";

    private String labelPrefix;

    private TrainingResult trainingResult;

    /**
     * Constructs a {@link FtTextClassification}.
     *
     * @param fta the {@link FtWrapper} containing the "fasttext model"
     * @param labelPrefix the prefix to use for labels
     */
    public FtTextClassification(FtWrapper fta, String labelPrefix) {
        super(fta);
        this.labelPrefix = labelPrefix;
    }

    /**
     * Trains the fastText model.
     *
     * @param config the training configuration to use
     * @param dataset the training dataset
     * @return the result of the training
     * @throws IOException when IO operation fails in loading a resource
     */
    public static FtTextClassification fit(FtTrainingConfig config, RawDataset<Path> dataset)
            throws IOException {
        Path outputDir = config.getOutputDir();
        if (Files.notExists(outputDir)) {
            Files.createDirectory(outputDir);
        }
        String fitModelName = config.getModelName();
        FtWrapper fta = FtWrapper.newInstance();
        Path modelFile = outputDir.resolve(fitModelName).toAbsolutePath();

        String[] args = config.toCommand(dataset.getData().toString());

        fta.runCmd(args);

        TrainingResult result = new TrainingResult();
        int epoch = config.getEpoch();
        if (epoch <= 0) {
            epoch = 5;
        }
        result.setEpoch(epoch);

        FtTextClassification block = new FtTextClassification(fta, config.getLabelPrefix());
        block.modelFile = modelFile;
        block.trainingResult = result;
        return block;
    }

    /**
     * Returns the fasttext label prefix.
     *
     * @return the fasttext label prefix
     */
    public String getLabelPrefix() {
        return labelPrefix;
    }

    /**
     * Returns the results of training, or null if not trained.
     *
     * @return the results of training, or null if not trained
     */
    public TrainingResult getTrainingResult() {
        return trainingResult;
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        PassthroughNDArray inputWrapper = (PassthroughNDArray) inputs.singletonOrThrow();
        String input = (String) inputWrapper.getObject();
        Classifications result = fta.predictProba(input, -1, labelPrefix);
        return new NDList(new PassthroughNDArray(result));
    }

    /**
     * Converts the block into the equivalent {@link FtWordEmbeddingBlock}.
     *
     * @return the equivalent {@link FtWordEmbeddingBlock}
     */
    public FtWordEmbeddingBlock toWordEmbedding() {
        return new FtWordEmbeddingBlock(fta);
    }

    /**
     * Returns the classifications of the input text.
     *
     * @param text the input text to be classified
     * @return classifications of the input text
     */
    public Classifications classify(String text) {
        return classify(text, -1);
    }

    /**
     * Returns top K classifications of the input text.
     *
     * @param text the input text to be classified
     * @param topK the value of K
     * @return classifications of the input text
     */
    public Classifications classify(String text, int topK) {
        return fta.predictProba(text, topK, labelPrefix);
    }
}
