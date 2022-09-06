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
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.nn.Blocks;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class NoopServingTranslatorFactoryTest {

    @Test
    public void testNoopTranslatorFactory() throws ModelException, IOException {
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

        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            Assert.assertNotNull(model);
        }
    }
}
