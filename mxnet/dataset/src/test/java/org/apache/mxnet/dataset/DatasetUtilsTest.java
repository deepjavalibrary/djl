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
package org.apache.mxnet.dataset;

import org.testng.Assert;
import org.testng.annotations.Test;
import software.amazon.ai.Device;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.dataset.Batch;

public class DatasetUtilsTest {

    @Test
    public void testSplitEven() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Device[] devices = new Device[] {Device.cpu(0), Device.cpu(1), Device.cpu(2)};

            NDArray data = manager.randomUniform(0, 1, new Shape(6, 5, 5, 3));
            NDArray label = manager.zeros(new Shape(6, 1));
            Batch record = new Batch(new NDList(data), new NDList(label));

            Batch[] split = DatasetUtils.split(record, devices, true);

            Assert.assertEquals(split.length, devices.length);

            int step = 2;
            for (int i = 0; i < split.length; i++) {
                NDArray array = split[i].getData().head();
                Assert.assertEquals(array.getDevice(), devices[i]);

                Assertions.assertEquals(
                        data.get(String.format("%d:%d", i * step, (i + 1) * step))
                                .asInDevice(devices[i], true),
                        array);
            }
        }
    }

    @Test
    public void testSplitUnevenData() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Device[] devices = new Device[] {Device.cpu(0), Device.cpu(1), Device.cpu(2)};

            NDArray data = manager.randomUniform(0, 1, new Shape(7, 5, 5, 3));
            NDArray label = manager.zeros(new Shape(7, 1));
            Batch record = new Batch(new NDList(data), new NDList(label));

            Batch[] split = DatasetUtils.split(record, devices, false);

            Assert.assertEquals(split.length, devices.length);

            int step = 2;
            for (int i = 0; i < split.length; i++) {
                NDArray array = split[i].getData().head();
                Assert.assertEquals(array.getDevice(), devices[i]);
                if (i == split.length - 1) {
                    Assertions.assertEquals(
                            data.get(String.format("%d:%d", i * step, data.size(0)))
                                    .asInDevice(devices[i], true),
                            array);
                    return;
                }
                Assertions.assertEquals(
                        data.get(String.format("%d:%d", i * step, (i + 1) * step))
                                .asInDevice(devices[i], true),
                        array);
            }
        }
    }

    @Test
    public void testSplitSmailerBatchSize() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Device[] devices = new Device[] {Device.cpu(0), Device.cpu(1), Device.cpu(2)};

            NDArray data = manager.randomUniform(0, 1, new Shape(2, 5, 5, 3));
            NDArray label = manager.zeros(new Shape(7, 1));
            Batch record = new Batch(new NDList(data), new NDList(label));

            Batch[] split = DatasetUtils.split(record, devices, false);

            Assert.assertEquals(split.length, 2);

            int step = 1;
            for (int i = 0; i < split.length; i++) {
                NDArray array = split[i].getData().head();
                Assert.assertEquals(array.getDevice(), devices[i]);
                if (i == split.length - 1) {
                    Assertions.assertEquals(
                            data.get(String.format("%d:%d", i * step, data.size(0)))
                                    .asInDevice(devices[i], true),
                            array);
                    return;
                }
                Assertions.assertEquals(
                        data.get(String.format("%d:%d", i * step, (i + 1) * step))
                                .asInDevice(devices[i], true),
                        array);
            }
        }
    }
}
