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

import ai.djl.training.TrainingListener;

public interface ExampleTrainingListeners {

    static TrainingListener[] exampleListeners(
            int batchSize, int trainDataSize, int validateDataSize, String outputDir) {
        return new TrainingListener[] {
            new EpochTrainingListener(),
            new MemoryTrainingListener(outputDir),
            new DivergenceCheckTrainingListener(),
            new EvaluatorMetricsTrainingListener(),
            new LoggingTrainingListener(batchSize, trainDataSize, validateDataSize),
            new TimeMeasureTrainingListener(outputDir)
        };
    }
}
