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
            data.put("instance", list);
            in.add(BytesSupplier.wrapAsJson(data));
            Output out = predictor.predict(in);
            BytesSupplier outData = out.getData();
            Assert.assertEquals(outData.getAsString(), "{\"predictions\":[[1.0,0.1],[2.0,0.2]]}");
        }
    }
}
