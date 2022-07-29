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

import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;

import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;

/* Integration test to check if model and inference runs fine */
public class ModelLoadingTest {

    @Test
    public void testModelLoading() throws IOException, ModelException, TranslateException {

        String modelUrl = "https://resources.djl.ai/demo/pytorch/traced_resnet18.zip";
        ImageClassificationTranslator.builder()
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .optApplySoftmax(true)
                .build();
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelUrls(modelUrl)
                        .optTranslator(
                                ImageClassificationTranslator.builder()
                                        .addTransform(new Resize(224, 224))
                                        .addTransform(new ToTensor())
                                        .optApplySoftmax(true)
                                        .build())
                        .build();
        Image image =
                ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/kitten.jpg");
        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Classifications result = predictor.predict(image);
            Assert.assertEquals("n02124075 Egyptian cat", result.best().getClassName());
        }
    }

    @Test
    public void testONNXRuntimeModelLoading() throws IOException, ModelException, TranslateException{
        String modelUrl = "https://mlrepo.djl.ai/model/tabular/softmax_regression/ai/djl/onnxruntime/iris_flowers/0.0.1/iris_flowers.zip";
        Criteria<IrisFlower, Classifications> criteria = Criteria.builder()
                .setTypes(IrisFlower.class, Classifications.class)
                .optModelUrls(modelUrl)
                .optTranslator(new MyTranslator())
                .optEngine("OnnxRuntime") // use OnnxRuntime engine by default
                .build();

        ZooModel<IrisFlower, Classifications> model = criteria.loadModel();
        Predictor<IrisFlower, Classifications> predictor = model.newPredictor();
        IrisFlower info = new IrisFlower(1.0f, 2.0f, 3.0f, 4.0f);
        Classifications result = predictor.predict(info);
        Assert.assertEquals("virginica", result.best().getClassName());
    }
}

class IrisFlower {

    public float sepalLength;
    public float sepalWidth;
    public float petalLength;
    public float petalWidth;

    public IrisFlower(float sepalLength, float sepalWidth, float petalLength, float petalWidth) {
        this.sepalLength = sepalLength;
        this.sepalWidth = sepalWidth;
        this.petalLength = petalLength;
        this.petalWidth = petalWidth;
    }
}


class MyTranslator implements NoBatchifyTranslator<IrisFlower, Classifications> {

    private final List<String> synset;

    public MyTranslator() {
        // species name
        synset = Arrays.asList("setosa", "versicolor", "virginica");
    }

    @Override
    public NDList processInput(TranslatorContext ctx, IrisFlower input) {
        float[] data = {input.sepalLength, input.sepalWidth, input.petalLength, input.petalWidth};
        NDArray array = ctx.getNDManager().create(data, new Shape(1, 4));
        return new NDList(array);
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        float[] data = list.get(1).toFloatArray();
        List<Double> probabilities = new ArrayList<>(data.length);
        for (float f : data) {
            probabilities.add((double) f);
        }
        return new Classifications(synset, probabilities);
    }
}