/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.util;

import org.testng.Assert;
import org.testng.annotations.Test;

public class Float16UtilsTest {

    @Test
    public void testFixedValues() {
        float[] from =
                new float[] {
                    0.44700000f,
                    0.553f,
                    0.999f,
                    0.9998f,
                    -0.9998f,
                    4.782f,
                    0.3f,
                    0.2f,
                    0.002f,
                    1f,
                    2f,
                    -2f,
                    0.22f,
                    0.44f
                };
        for (float v : from) {
            float value = (float) Math.random() * 2048f;
            float found = Float16Utils.halfToFloat(Float16Utils.floatToHalf(value));
            float diff = Math.abs(found - value);
            Assert.assertTrue(diff < 0.6);
        }
    }

    @Test
    public void testRandomValues() {
        for (int i = 0; i < 2048 * 10; i++) {
            float value = (float) Math.random() * 2048f;
            float found = Float16Utils.halfToFloat(Float16Utils.floatToHalf(value));
            float diff = Math.abs(found - value);
            Assert.assertTrue(diff < 0.6);
        }
    }
}
