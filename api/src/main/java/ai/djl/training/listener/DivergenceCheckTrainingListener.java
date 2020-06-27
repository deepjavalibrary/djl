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
package ai.djl.training.listener;

import ai.djl.TrainingDivergedException;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;

/** {@link TrainingListener} that gives early warning if your training has failed by divergence. */
public class DivergenceCheckTrainingListener extends TrainingListenerAdapter {

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        Loss trainingLoss = trainer.getLoss();
        if (Float.isNaN(trainingLoss.getAccumulator(EvaluatorTrainingListener.TRAIN_ALL))) {
            throw new TrainingDivergedException(
                    "The Loss became NaN, try reduce learning rate,"
                            + "add clipGradient option to your optimizer, check input data and loss calculation.");
        }
    }
}
