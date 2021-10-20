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
package ai.djl.modality;

import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.nio.charset.StandardCharsets;
import org.testng.Assert;
import org.testng.annotations.Test;

public class InputOutputTest {

    @Test
    public void testOutput() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Output output = new Output();
            output.addProperty("content-type", "tensor/ndlist");
            output.add("bytes".getBytes(StandardCharsets.UTF_8));
            output.add("str");
            output.add(manager.ones(new Shape(1)));
            output.add(new NDList(manager.ones(new Shape(2))));
            output.add("input_1", "str1");
            output.add("input_2", "bytes1".getBytes(StandardCharsets.UTF_8));
            output.add("input_3", manager.ones(new Shape(3)));
            output.add("input_4", new NDList(manager.zeros(new Shape(2))));
            output.add(1, "data", new NDList(manager.zeros(new Shape(1))));

            Assert.assertEquals(output.getProperty("Content-Type", null), "tensor/ndlist");

            // non-exists key
            Assert.assertNull(output.get("null"));
            Assert.assertNull(output.getAsString("null"));
            Assert.assertNull(output.getAsBytes("null"));
            Assert.assertNull(output.getAsNDArray(manager, "null"));
            Assert.assertNull(output.getAsNDList(manager, "null"));

            Assert.assertEquals(output.getAsBytes(2), "str".getBytes(StandardCharsets.UTF_8));
            Assert.assertEquals(output.getAsString(0), "bytes");
            Assert.assertEquals(
                    output.getAsBytes("input_1"), "str1".getBytes(StandardCharsets.UTF_8));
            Assert.assertEquals(output.getAsString("input_1"), "str1");
            Shape shape = output.getAsNDArray(manager, "input_3").getShape();
            Assert.assertEquals(shape.get(0), 3);
            shape = output.getAsNDList(manager, "input_3").head().getShape();
            Assert.assertEquals(shape.get(0), 3);
            Assert.assertEquals(output.getAsBytes("data").length, 58);
            Assert.assertEquals(output.getAsBytes("input_3").length, 12);

            BytesSupplier data = output.getData();
            Assert.assertEquals(data, output.getDataAsNDList(manager));

            data = output.get(3);
            Assert.assertTrue(data instanceof NDArray);

            output = new Output();
            output.add(manager.zeros(new Shape(1)));
            NDList ndlist = output.getDataAsNDList(manager);
            Assert.assertEquals(ndlist.size(), 1);
        }
    }
}
