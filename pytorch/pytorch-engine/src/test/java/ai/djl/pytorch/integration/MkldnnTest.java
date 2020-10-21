/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.pytorch.integration;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.Arrays;
import org.testng.annotations.Test;

/** The file is for testing PyTorch MKLDNN functionalities. */
public class MkldnnTest {

    @Test
    public void testMkldnn() {
        System.setProperty("ai.djl.pytorch.use_mkldnn", "true");
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray[] arrays = {
                manager.create(new float[] {0f, 1f, 3f, 4f}, new Shape(2, 2)),
                manager.zeros(new Shape(2, 2)),
                manager.ones(new Shape(2, 2)),
                manager.ones(new Shape(2, 2)).duplicate(),
                manager.full(new Shape(2, 2), 1f),
                manager.zeros(new Shape(2, 2)).zerosLike(),
                manager.zeros(new Shape(2, 2)).onesLike(),
                manager.eye(2),
                manager.randomNormal(new Shape(2, 2)),
                manager.randomUniform(0, 1, new Shape(2, 2))
            };
            // run sanity check, if two arrays are on different layout, it will throw exception
            Arrays.stream(arrays).reduce(NDArray::add);
            Arrays.stream(arrays).forEach(NDArray::toString);
            arrays = new NDArray[] {manager.arange(4f), manager.linspace(0, 1, 4)};
            // run sanity check, if two arrays are on different layout, it will throw exception
            Arrays.stream(arrays).reduce(NDArray::add);
            Arrays.stream(arrays).forEach(NDArray::toString);
        } finally {
            System.setProperty("ai.djl.pytorch.use_mkldnn", "false");
        }
    }
}
