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

import org.junit.Assert;
import org.junit.Test;
import software.amazon.ai.Context;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;

public class DatasetUtilsTest {
    @Test
    public void testSplitData() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.randomUniform(0, 1, new Shape(10, 5, 5, 3));
            NDList list = DatasetUtils.splitData(data, 5, 0, true);
            // normal case
            int step = 2;
            for (int i = 0; i < list.size(); i++) {
                Assertions.assertEquals(
                        data.get(String.format("%d:%d", i * step, (i + 1) * step)), list.get(i));
            }
            // uneven data case
            list = DatasetUtils.splitData(data, 4, 0, false);
            for (int i = 0; i < list.size(); i++) {
                if (i == list.size() - 1) {
                    Assertions.assertEquals(
                            data.get(String.format("%d:%d", i * step, data.size(0))), list.get(i));
                    break;
                }
                Assertions.assertEquals(
                        data.get(String.format("%d:%d", i * step, (i + 1) * step)), list.get(i));
            }
            // numOfSlice is greater than size
            list = DatasetUtils.splitData(data, 15, 0, false);
            step = 1;
            for (int i = 0; i < list.size(); i++) {
                Assertions.assertEquals(
                        data.get(String.format("%d:%d", i * step, (i + 1) * step)), list.get(i));
            }
            // splitAxis is not 0
            list = DatasetUtils.splitData(data, 2, 2, false);
            step = 2;
            for (int i = 0; i < list.size(); i++) {
                if (i == list.size() - 1) {
                    Assertions.assertEquals(
                            data.get(String.format(":,:,%d:%d,:", i * step, data.size(2))),
                            list.get(i));
                    break;
                }
                Assertions.assertEquals(
                        data.get(String.format(":,:,%d:%d,:", i * step, (i + 1) * step)),
                        list.get(i));
            }
        }
    }

    @Test
    public void testSplitAndLoad() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.randomUniform(0, 1, new Shape(6, 5, 5, 3));
            Context[] contexts = new Context[] {Context.cpu(0), Context.cpu(1), Context.cpu(2)};
            NDList list = DatasetUtils.splitAndLoad(data, contexts, true);
            int step = 2;
            for (int i = 0; i < list.size(); i++) {
                Assertions.assertEquals(
                        data.get(String.format("%d:%d", i * step, (i + 1) * step)), list.get(i));
                Assert.assertEquals(list.get(i).getContext(), contexts[i]);
            }
            // uneven data case
            data = manager.randomUniform(0, 1, new Shape(7, 5, 5, 3));
            list = DatasetUtils.splitAndLoad(data, contexts, false);
            for (int i = 0; i < list.size(); i++) {
                Assert.assertEquals(list.get(i).getContext(), contexts[i]);
                if (i == list.size() - 1) {
                    Assertions.assertEquals(
                            data.get(String.format("%d:%d", i * step, data.size(0))), list.get(i));
                    break;
                }
                Assertions.assertEquals(
                        data.get(String.format("%d:%d", i * step, (i + 1) * step)), list.get(i));
            }
            // numOfSlice is greater than size
            data = manager.randomUniform(0, 1, new Shape(2, 5, 5, 3));
            list = DatasetUtils.splitAndLoad(data, contexts, 0, false);
            step = 1;
            for (int i = 0; i < list.size(); i++) {
                Assert.assertEquals(list.get(i).getContext(), contexts[i]);
                Assertions.assertEquals(
                        data.get(String.format("%d:%d", i * step, (i + 1) * step)), list.get(i));
            }
            // splitAxis is not 0
            data = manager.randomUniform(0, 1, new Shape(3, 5, 5, 3));
            list = DatasetUtils.splitAndLoad(data, contexts, 2, false);
            step = 1;
            for (int i = 0; i < list.size(); i++) {
                Assert.assertEquals(list.get(i).getContext(), contexts[i]);
                if (i == list.size() - 1) {
                    Assertions.assertEquals(
                            data.get(String.format(":,:,%d:%d,:", i * step, data.size(2))),
                            list.get(i));
                    break;
                }
                Assertions.assertEquals(
                        data.get(String.format(":,:,%d:%d,:", i * step, (i + 1) * step)),
                        list.get(i));
            }
        }
    }
}
