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
package ai.djl.android.core;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.onnxruntime.zoo.tabular.softmax_regression.IrisClassificationTranslatorFactory;
import ai.djl.onnxruntime.zoo.tabular.softmax_regression.IrisFlower;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;

/* Integration test to check if model and inference runs fine */
public class ModelLoadingTest {

    @Test
    public void testModelLoading() throws IOException, ModelException, TranslateException {
        String modelUrl = "https://resources.djl.ai/demo/pytorch/traced_resnet18.zip";
        ImageClassificationTranslator translator =
                ImageClassificationTranslator.builder()
                        .addTransform(new Resize(224, 224))
                        .addTransform(new ToTensor())
                        .optApplySoftmax(true)
                        .build();
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelUrls(modelUrl)
                        .optTranslator(translator)
                        .build();

        ImageFactory factory = ImageFactory.getInstance();
        Image image = factory.fromUrl("https://resources.djl.ai/images/kitten.jpg");
        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Classifications result = predictor.predict(image);
            Assert.assertEquals("n02124075 Egyptian cat", result.best().getClassName());
        }
    }

    @Test
    public void testONNXRuntimeModelLoading()
            throws IOException, ModelException, TranslateException {
        String modelUrl =
                "https://mlrepo.djl.ai/model/tabular/softmax_regression/ai/djl/onnxruntime/iris_flowers/0.0.1/iris_flowers.zip";
        Criteria<IrisFlower, Classifications> criteria =
                Criteria.builder()
                        .setTypes(IrisFlower.class, Classifications.class)
                        .optModelUrls(modelUrl)
                        .optTranslatorFactory(new IrisClassificationTranslatorFactory())
                        .optEngine("OnnxRuntime") // use OnnxRuntime engine by default
                        .build();

        ZooModel<IrisFlower, Classifications> model = criteria.loadModel();
        Predictor<IrisFlower, Classifications> predictor = model.newPredictor();
        IrisFlower info = new IrisFlower(1.0f, 2.0f, 3.0f, 4.0f);
        Classifications result = predictor.predict(info);
        Assert.assertEquals("virginica", result.best().getClassName());
    }
}
