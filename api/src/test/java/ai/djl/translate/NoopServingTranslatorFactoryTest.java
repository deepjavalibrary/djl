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
package ai.djl.translate;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.nn.Blocks;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class NoopServingTranslatorFactoryTest {

    @Test
    public void testNoopTranslatorFactory() throws ModelException, IOException, TranslateException {
        NoopServingTranslatorFactory factory = new NoopServingTranslatorFactory();
        Assert.assertEquals(factory.getSupportedTypes().size(), 1);

        Path modelPath = Paths.get("src/test/resources/identity");
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelPath)
                        .optBlock(Blocks.identityBlock())
                        .optTranslatorFactory(factory)
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {

            Input in = new Input();
            in.addProperty("Content-Type", "application/json; charset=UTF-8");
            Map<String, List<List<Number>>> data = new ConcurrentHashMap<>();
            List<List<Number>> list = new ArrayList<>();
            list.add(Arrays.asList(1.0f, 0.1f));
            list.add(Arrays.asList(2.0f, 0.2f));
            data.put("instances", list);
            in.add(BytesSupplier.wrapAsJson(data));
            Output out = predictor.predict(in);
            BytesSupplier outData = out.getData();
            Assert.assertEquals(outData.getAsString(), "{\"predictions\":[[1.0,0.1],[2.0,0.2]]}");

            // CSV input / JSON output
            Input csvJsonIn = new Input();
            csvJsonIn.addProperty("Content-Type", "text/csv");
            csvJsonIn.addProperty("Accept", "application/json");
            csvJsonIn.add("1.0,0.1\n2.0,0.2\n");
            Output csvJsonOut = predictor.predict(csvJsonIn);
            Assert.assertEquals(
                    csvJsonOut.getData().getAsString(), "{\"predictions\":[[1.0,0.1],[2.0,0.2]]}");

            // CSV input / CSV output
            Input csvCsvIn = new Input();
            csvCsvIn.addProperty("Content-Type", "text/csv");
            csvCsvIn.addProperty("Accept", "text/csv");
            csvCsvIn.add("1.0,0.1\n2.0,0.2\n");
            Output csvCsvOut = predictor.predict(csvCsvIn);
            Assert.assertEquals(csvCsvOut.getData().getAsString(), "1.0,0.1\n2.0,0.2\n");

            // Uneven rows should fail
            Input unevenRowsIn = new Input();
            unevenRowsIn.addProperty("Content-Type", "text/csv");
            unevenRowsIn.add("1.0,0.1\n2.0,0.2,0.3\n");
            try {
                predictor.predict(unevenRowsIn);
                Assert.fail("Should have thrown exception for uneven rows");
            } catch (Exception e) {
                String msg = e.getMessage();
                if (e.getCause() != null) {
                    msg += " | cause: " + e.getCause().getMessage();
                }
                Assert.assertTrue(
                        msg.contains("columns, expected"), "Unexpected exception message: " + msg);
            }

            // Non-numeric should fail
            Input nonNumericIn = new Input();
            nonNumericIn.addProperty("Content-Type", "text/csv");
            nonNumericIn.add("1.0,hello\n2.0,0.2\n");
            try {
                predictor.predict(nonNumericIn);
                Assert.fail("Should have thrown exception for non-numeric data");
            } catch (Exception e) {
                String msg = e.getMessage();
                if (e.getCause() != null) {
                    msg += " | cause: " + e.getCause().getMessage();
                }
                Assert.assertTrue(
                        msg.contains("Non-numeric"), "Unexpected exception message: " + msg);
            }

            // Header row should be skipped
            Input headerIn = new Input();
            headerIn.addProperty("Content-Type", "text/csv");
            headerIn.addProperty("Accept", "application/json");
            headerIn.add("feature1,feature2\n1.0,0.1\n2.0,0.2\n");
            Output headerOut = predictor.predict(headerIn);
            Assert.assertEquals(
                    headerOut.getData().getAsString(), "{\"predictions\":[[1.0,0.1],[2.0,0.2]]}");

            // Empty CSV should fail
            Input emptyIn = new Input();
            emptyIn.addProperty("Content-Type", "text/csv");
            emptyIn.add("");
            try {
                predictor.predict(emptyIn);
                Assert.fail("Should have thrown exception for empty CSV");
            } catch (Exception e) {
                String msg = e.getMessage();
                if (e.getCause() != null) {
                    msg += " | cause: " + e.getCause().getMessage();
                }
                Assert.assertTrue(
                        msg.toLowerCase().contains("csv") && msg.toLowerCase().contains("empty"),
                        "Unexpected exception message: " + msg);
            }
        }
    }
}
