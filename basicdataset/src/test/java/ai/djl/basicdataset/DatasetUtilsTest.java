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
package ai.djl.basicdataset;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.Batchifier;
import java.util.Locale;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DatasetUtilsTest {

    @Test
    public void testSplitEven() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Device[] devices = new Device[] {Device.cpu(), Device.cpu(), Device.cpu()};

            NDArray data = manager.randomUniform(0, 1, new Shape(6, 5, 5, 3));
            NDArray label = manager.zeros(new Shape(6, 1));
            Batch batch =
                    new Batch(
                            manager.newSubManager(),
                            new NDList(data),
                            new NDList(label),
                            6,
                            Batchifier.STACK,
                            Batchifier.STACK,
                            0,
                            0);

            Batch[] split = batch.split(devices, true);

            Assert.assertEquals(split.length, devices.length);

            int step = 2;
            for (int i = 0; i < split.length; i++) {
                NDArray array = split[i].getData().singletonOrThrow();
                Assert.assertEquals(array.getDevice(), devices[i]);

                String ndIndex = String.format(Locale.ROOT, "%d:%d", i * step, (i + 1) * step);
                Assert.assertEquals(data.get(ndIndex).toDevice(devices[i], true), array);
            }
        }
    }

    @Test
    public void testSplitUnevenData() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Device[] devices = new Device[] {Device.cpu(), Device.cpu(), Device.cpu()};

            NDArray data = manager.randomUniform(0, 1, new Shape(7, 5, 5, 3));
            NDArray label = manager.zeros(new Shape(7, 1));
            Batch batch =
                    new Batch(
                            manager.newSubManager(),
                            new NDList(data),
                            new NDList(label),
                            7,
                            Batchifier.STACK,
                            Batchifier.STACK,
                            0,
                            0);

            Batch[] split = batch.split(devices, false);

            Assert.assertEquals(split.length, devices.length);

            int step = 3;
            for (int i = 0; i < split.length; i++) {
                NDArray array = split[i].getData().singletonOrThrow();
                Assert.assertEquals(array.getDevice(), devices[i]);
                if (i == split.length - 1) {
                    String indices = String.format(Locale.ROOT, "%d:%d", i * step, data.size(0));
                    Assert.assertEquals(data.get(indices).toDevice(devices[i], true), array);
                    return;
                }
                String indices = String.format(Locale.ROOT, "%d:%d", i * step, (i + 1) * step);
                Assert.assertEquals(data.get(indices).toDevice(devices[i], true), array);
            }
        }
    }

    @Test
    public void testSplitSmallerBatchSize() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Device[] devices = new Device[] {Device.cpu(), Device.cpu(), Device.cpu()};

            NDArray data = manager.randomUniform(0, 1, new Shape(2, 5, 5, 3));
            NDArray label = manager.zeros(new Shape(7, 1));
            Batch batch =
                    new Batch(
                            manager.newSubManager(),
                            new NDList(data),
                            new NDList(label),
                            7,
                            Batchifier.STACK,
                            Batchifier.STACK,
                            0,
                            0);

            Batch[] split = batch.split(devices, false);

            Assert.assertEquals(split.length, 2);

            int step = 1;
            for (int i = 0; i < split.length; i++) {
                NDArray array = split[i].getData().singletonOrThrow();
                Assert.assertEquals(array.getDevice(), devices[i]);
                if (i == split.length - 1) {
                    String indices = String.format(Locale.ROOT, "%d:%d", i * step, data.size(0));
                    Assert.assertEquals(data.get(indices).toDevice(devices[i], true), array);
                    return;
                }
                String indices = String.format(Locale.ROOT, "%d:%d", i * step, (i + 1) * step);
                Assert.assertEquals(data.get(indices).toDevice(devices[i], true), array);
            }
        }
    }

    @Test
    public void testStackBatchifierSplit() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDList data = new NDList(manager.randomUniform(0, 1, new Shape(7, 5, 5, 3)));
            Batchifier stackBatchifier = Batchifier.STACK;
            Batchifier defaultBatchifier =
                    new Batchifier() {
                        @Override
                        public NDList batchify(NDList[] inputs) {
                            return stackBatchifier.batchify(inputs);
                        }

                        /** {@inheritDoc} */
                        @Override
                        public NDList[] unbatchify(NDList inputs) {
                            return stackBatchifier.unbatchify(inputs);
                        }
                    };
            NDList[] fromStack = stackBatchifier.split(data, 3, false);
            NDList[] fromDef = defaultBatchifier.split(data, 3, false);
            for (int i = 0; i < 3; i++) {
                Assertions.assertAlmostEquals(
                        fromDef[i].singletonOrThrow(), fromStack[i].singletonOrThrow());
            }
        }
    }
}
