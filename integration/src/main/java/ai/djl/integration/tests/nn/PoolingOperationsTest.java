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
package ai.djl.integration.tests.nn;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PoolingOperationsTest {
    TrainingConfig config =
            new DefaultTrainingConfig(Loss.l2Loss())
                    .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

    @Test
    public void testMaxPool1d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.maxPool1dBlock(new Shape(2)));
            // Look for a max pool value 5
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 5);
                NDArray expected = manager.ones(new Shape(2, 2, 1));
                expected.set(new NDIndex(1, 1, 0), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testMaxPool2d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.maxPool2dBlock(new Shape(2, 2)));
            // Look for a max pool value 5
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1), 5);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1));
                expected.set(new NDIndex(1, 1, 0, 0), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testMaxPool3d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.maxPool3dBlock(new Shape(2, 2, 2)));
            // Look for a max pool value 5
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1, 1), 5);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1, 1));
                expected.set(new NDIndex(1, 1, 0, 0, 0), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalMaxPool1d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalMaxPool1dBlock());
            // Look for a max pool value 5
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 5);
                NDArray expected = manager.ones(new Shape(2, 2));
                expected.set(new NDIndex(1, 1), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalMaxPool2d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalMaxPool2dBlock());
            // Look for a max pool value 5
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1), 5);
                NDArray expected = manager.ones(new Shape(2, 2));
                expected.set(new NDIndex(1, 1), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalMaxPool3d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalMaxPool3dBlock());
            // Look for a max pool value 5
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1, 1), 5);
                NDArray expected = manager.ones(new Shape(2, 2));
                expected.set(new NDIndex(1, 1), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testAvgPool1d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.avgPool1dBlock(new Shape(2)));
            // Look for a average pool value 1.5
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1));
                expected.set(new NDIndex(1, 1, 0), 1.5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testAvgPool2d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.avgPool2dBlock(new Shape(2, 2)));
            // Look for a average pool value 1.25
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1));
                expected.set(new NDIndex(1, 1, 0, 0), 1.25);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testAvgPool3d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.avgPool3dBlock(new Shape(2, 2, 2)));
            // Look for a average pool value 1.125
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1, 1));
                expected.set(new NDIndex(1, 1, 0, 0, 0), 1.125);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalAvgPool1d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalAvgPool1dBlock());
            // Look for a average pool value 1.5
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2));
                expected.set(new NDIndex(1, 1), 1.5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalAvgPool2d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalAvgPool2dBlock());
            // Look for a average pool value 1.5
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2));
                expected.set(new NDIndex(1, 1), 1.25);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalAvgPool3d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalAvgPool3dBlock());
            // Look for a average pool value 1.5
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2));
                expected.set(new NDIndex(1, 1), 1.125);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testLpPool1d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.lpPool1dBlock(1, new Shape(2)));
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1));
                expected.muli(2);
                expected.set(new NDIndex(1, 1, 0), 3);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testLpPool2d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.lpPool2dBlock(1, new Shape(2, 2)));
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1));
                expected.muli(4);
                expected.set(new NDIndex(1, 1, 0, 0), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    // lpPool3d is not supported in PyTorch engine
    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testLpPool3d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.lpPool3dBlock(1, new Shape(2, 2, 2)));
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2, 1, 1, 1));
                expected.muli(8);
                expected.set(new NDIndex(1, 1, 0, 0, 0), 9);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalLpPool1d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalLpPool1dBlock(1));
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2));
                data.set(new NDIndex(1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2));
                expected.muli(2);
                expected.set(new NDIndex(1, 1), 3);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalLpPool2d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalLpPool2dBlock(1));
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2));
                expected.muli(4);
                expected.set(new NDIndex(1, 1), 5);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGlobalLpPool3d() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Pool.globalLpPool3dBlock(1));
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 2, 2, 2, 2));

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(2, 2, 2, 2, 2));
                data.set(new NDIndex(1, 1, 1, 1, 1), 2);
                NDArray expected = manager.ones(new Shape(2, 2));
                expected.muli(8);
                expected.set(new NDIndex(1, 1), 9);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }
}
