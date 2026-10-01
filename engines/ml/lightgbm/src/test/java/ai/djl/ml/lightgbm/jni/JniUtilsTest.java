/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ml.lightgbm.jni;

import ai.djl.engine.EngineException;

import org.testng.Assert;
import org.testng.annotations.Test;

public class JniUtilsTest {

    /**
     * The prediction output buffer length was previously computed as {@code classes * rows *
     * iterations} in 32-bit {@code int}, which wraps for large models or inputs. The length is now
     * computed in 64-bit and rejected if it does not fit in an int.
     */
    @Test
    public void testBufferLengthIntegerOverflow() {
        Assert.assertEquals(JniUtils.toBufferLength(3, 10), 30);
        Assert.assertEquals(JniUtils.toBufferLength(3, 10, 5), 150);

        // 3 * 2^30 == 3221225472, which wraps int32 to a negative value.
        Assert.assertThrows(ArithmeticException.class, () -> JniUtils.toBufferLength(3, 1 << 30));

        // 2^16 * 2^16 * 2 == 2^33, which wraps int32 to exactly 0.
        Assert.assertThrows(
                ArithmeticException.class, () -> JniUtils.toBufferLength(1 << 16, 1 << 16, 2));

        Assert.assertThrows(IllegalArgumentException.class, () -> JniUtils.toBufferLength(-1, 10));
    }

    /**
     * The buffer lengths assume one output value per class for each row and iteration, so a model
     * whose number of trees per iteration differs from its number of classes is rejected when it is
     * loaded.
     */
    @Test
    public void testTreesPerIterationMismatch() {
        JniUtils.checkTreesPerIteration(1, 1);
        JniUtils.checkTreesPerIteration(3, 3);

        EngineException e =
                Assert.expectThrows(
                        EngineException.class, () -> JniUtils.checkTreesPerIteration(1, 2));
        Assert.assertEquals(
                e.getMessage(), "Invalid LightGBM model: 1 classes but 2 trees per iteration");
    }
}
