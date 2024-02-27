/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class TranslatorTest {

    @Test
    public void testBatchTranslator() throws IOException, ModelException, TranslateException {
        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            NDArray arr = manager.ones(new Shape(2, 1000));
                            return new NDList(arr);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        ImageFactory factory = ImageFactory.getInstance();
        Image[] inputs = {
            factory.fromFile(Paths.get("../examples/src/test/resources/kitten.jpg")),
            factory.fromFile(Paths.get("../examples/src/test/resources/dog-cat.jpg"))
        };
        String[] classes = new String[1000];
        Arrays.fill(classes, "something");

        ImageClassificationTranslator translator =
                ImageClassificationTranslator.builder()
                        .addTransform(new Resize(256))
                        .addTransform(new CenterCrop(224, 224))
                        .addTransform(new ToTensor())
                        .optSynset(Arrays.asList(classes))
                        .build();

        Criteria<Image[], Classifications[]> criteria =
                Criteria.builder()
                        .setTypes(Image[].class, Classifications[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optOption("hasParameter", "false")
                        .optTranslator(translator.toBatchTranslator())
                        .build();

        try (ZooModel<Image[], Classifications[]> model = criteria.loadModel();
                Predictor<Image[], Classifications[]> predictor = model.newPredictor()) {
            Classifications[] res = predictor.predict(inputs);
            Assert.assertEquals(res.length, 2);
            int intValue = model.intProperty("something", -1);
            Assert.assertEquals(intValue, -1);
            long longValue = model.longProperty("something", -1L);
            Assert.assertEquals(longValue, -1L);
        }
    }
}
