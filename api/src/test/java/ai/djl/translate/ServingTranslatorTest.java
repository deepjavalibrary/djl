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

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ServingTranslatorTest {

    @AfterClass
    public void tierDown() {
        Utils.deleteQuietly(Paths.get("build/model"));
    }

    @Test
    public void testNumpy() throws IOException, TranslateException, ModelException {
        Path path = Paths.get("build/model");
        Files.createDirectories(path);
        Input input = new Input();

        try (NDManager manager = NDManager.newBaseManager()) {
            Block block = Blocks.identityBlock();
            block.initialize(manager, DataType.FLOAT32, new Shape(1));
            Model model = Model.newInstance("identity");
            model.setBlock(block);
            model.save(path, null);
            model.close();
            NDList list = new NDList();
            list.add(manager.create(10f));
            input.add(list.encode(true));
            input.add("Content-Type", "tensor/npz");
        }

        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(path)
                        .optModelName("identity")
                        .optBlock(Blocks.identityBlock())
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Output output = predictor.predict(input);
            try (NDManager manager = NDManager.newBaseManager()) {
                NDList list = output.getDataAsNDList(manager);
                Assert.assertEquals(list.size(), 1);
                Assert.assertEquals(list.get(0).toFloatArray()[0], 10f);
            }
            Input invalid = new Input();
            invalid.add("String");
            Assert.assertThrows(TranslateException.class, () -> predictor.predict(invalid));
        }
    }
}
