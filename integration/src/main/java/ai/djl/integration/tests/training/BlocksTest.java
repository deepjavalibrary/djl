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
package ai.djl.integration.tests.training;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import org.testng.annotations.Test;

public class BlocksTest {

    TrainingConfig config =
            new DefaultTrainingConfig(Loss.l2Loss())
                    .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

    @Test
    public void testFlattenBlock() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.batchFlattenBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                ParameterStore parameterStore = new ParameterStore(manager, false);

                NDArray data = manager.randomUniform(0, 255, new Shape(10, 28, 28));
                NDArray expected = data.reshape(10, 28 * 28);
                NDArray result =
                        model.getBlock().forward(parameterStore, new NDList(data), true).head();
                Assertions.assertAlmostEquals(result, expected);
            }
        }
    }
}
