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
package ai.djl.paddlepaddle.zoo;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ImageClassificationTest {

    @Test
    public void testImageClassification()
            throws MalformedModelException, ModelNotFoundException, IOException,
                    TranslateException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(
                                "http://paddle-inference-dist.bj.bcebos.com/resnet50_model.tar.gz")
                        .optModelName("model")
                        .optEngine("PaddlePaddle")
                        .optProgress(new ProgressBar())
                        .build();

        ZooModel<NDList, NDList> model = ModelZoo.loadModel(criteria);
        Predictor<NDList, NDList> predictor = model.newPredictor();
        NDManager manager = model.getNDManager();
        NDArray array = manager.ones(new Shape(1, 3, 300, 300));
        NDList output = predictor.predict(new NDList(array));
        Shape shape = output.singletonOrThrow().getShape();
        DataType dataType = output.singletonOrThrow().getDataType();
        Assert.assertEquals(shape, new Shape(1, 512));
        Assert.assertEquals(dataType, DataType.FLOAT32);
    }
}
