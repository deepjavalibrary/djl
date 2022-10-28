/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.timeseries.evaluator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.timeseries.distribution.output.NegativeBinomialOutput;

import org.testng.Assert;
import org.testng.annotations.Test;

public class RmsseTest {

    public static void main(String[] args) {
        RmsseTest t = new RmsseTest();
        t.testRmsse();
    }

    @Test
    public void testRmsse() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray totalCount = manager.create(new float[] {1f, 2f}).expandDims(0);
            NDArray logits = manager.create(new float[] {1f, 2f}).log().expandDims(0);
            totalCount.setName("total_count");
            logits.setName("logits");
            NDList prediction = new NDList(totalCount, logits);
            NDList label = new NDList(manager.create(new float[] {3f, 4f}).expandDims(0));

            Rmsse rmsse = new Rmsse(new NegativeBinomialOutput());
            rmsse.addAccumulator("");
            rmsse.updateAccumulator("", label, prediction);
            float rmsseValue = rmsse.getAccumulator("");
            float expectedRmsse = 1.414213562373095f;
            Assert.assertEquals(rmsseValue, expectedRmsse);
        }
    }
}
