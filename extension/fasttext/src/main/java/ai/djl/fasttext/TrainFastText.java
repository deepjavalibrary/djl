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
package ai.djl.fasttext;

import ai.djl.fasttext.zoo.nlp.textclassification.FtTextClassification;
import ai.djl.training.dataset.RawDataset;

import java.io.IOException;
import java.nio.file.Path;

/** A utility to aggregate options for training with fasttext. */
public final class TrainFastText {

    private TrainFastText() {}

    /**
     * Trains a fastText {@link ai.djl.Application.NLP#TEXT_CLASSIFICATION} model.
     *
     * @param config the training configuration to use
     * @param dataset the training dataset
     * @return the result of the training
     * @throws IOException when IO operation fails in loading a resource
     * @see FtTextClassification#fit(FtTrainingConfig, RawDataset)
     */
    public static FtTextClassification textClassification(
            FtTrainingConfig config, RawDataset<Path> dataset) throws IOException {
        return FtTextClassification.fit(config, dataset);
    }
}
