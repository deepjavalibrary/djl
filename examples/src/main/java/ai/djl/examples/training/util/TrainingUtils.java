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
package ai.djl.examples.training.util;

import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import java.io.IOException;

public final class TrainingUtils {

    private TrainingUtils() {}

    public static void fit(
            Trainer trainer,
            TrainingConfig config,
            Dataset trainingDataset,
            Dataset validateDataset)
            throws IOException {
        int numEpoch = config.getEpoch();

        for (int epoch = 0; epoch < numEpoch; epoch++) {
            trainer.resetTrainingMetrics();

            for (Batch batch : trainer.iterateDataset(trainingDataset)) {
                trainer.train(batch);
                trainer.step();
                batch.close();
            }

            if (validateDataset != null) {
                for (Batch batch : trainer.iterateDataset(validateDataset)) {
                    trainer.validate(batch);
                    batch.close();
                }
            }
        }
        trainer.resetTrainingMetrics();
    }
}
