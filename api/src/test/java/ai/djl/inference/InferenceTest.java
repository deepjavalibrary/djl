/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.inference;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.test.mock.EchoTranslator;
import ai.djl.test.mock.MockImageTranslator;
import ai.djl.test.mock.MockModel;
import ai.djl.test.mock.MockNDArray;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class InferenceTest {

    private Image image;

    @BeforeClass
    public void setup() throws IOException {
        Files.createDirectories(Paths.get("build/model"));
        image =
                ImageFactory.getInstance()
                        .fromImage(new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB));
    }

    @Test
    public void testObjectDection() throws IOException, TranslateException, ModelException {
        Path modelDir = Paths.get("build/model");
        String modelName = "mockModel";

        Model model = Model.newInstance(modelName);
        model.load(modelDir);
        MockImageTranslator translator = new MockImageTranslator("cat");

        Metrics metrics = new Metrics();
        try (Predictor<Image, DetectedObjects> ssd = model.newPredictor(translator)) {
            ssd.setMetrics(metrics);
            DetectedObjects result = ssd.predict(image);

            DetectedObjects.DetectedObject object = result.item(0);
            Assert.assertEquals(object.getClassName(), "cat");
            Assert.assertEquals(
                    Double.compare(object.getBoundingBox().getBounds().getHeight(), 1d), 0);
        }
    }

    @Test
    public void testClassifier() throws IOException, TranslateException, ModelException {
        Path modelDir = Paths.get("build/model");
        Model model = Model.newInstance("classifier");
        model.load(modelDir);

        final String className = "cat";
        Translator<String, Classifications> translator =
                new Translator<String, Classifications>() {

                    /** {@inheritDoc} */
                    @Override
                    public NDList processInput(TranslatorContext ctx, String input) {
                        return new NDList(
                                new MockNDArray(
                                        null,
                                        null,
                                        new Shape(3, 24, 24),
                                        DataType.FLOAT32,
                                        SparseFormat.DENSE));
                    }

                    /** {@inheritDoc} */
                    @Override
                    public Classifications processOutput(TranslatorContext ctx, NDList list) {
                        return new Classifications(
                                Collections.singletonList(className),
                                Collections.singletonList(0.9d));
                    }

                    /** {@inheritDoc} */
                    @Override
                    public Batchifier getBatchifier() {
                        return null;
                    }
                };
        Metrics metrics = new Metrics();

        try (Predictor<String, Classifications> classifier = model.newPredictor(translator)) {
            classifier.setMetrics(metrics);
            Classifications result = classifier.predict(className);
            Assert.assertEquals(Double.compare(result.get(className).getProbability(), 0.9d), 0);
        }
    }

    @Test(expectedExceptions = TranslateException.class)
    public void testTranslateException() throws TranslateException {
        EchoTranslator<String> translator = new EchoTranslator<>();
        translator.setInputException(new TranslateException("Some exception"));
        Model model = new MockModel();
        Predictor<String, String> predictor = model.newPredictor(translator);
        String result = predictor.predict("input");
        Assert.assertEquals(result, "input");
    }

    @Test(expectedExceptions = IOException.class)
    public void loadModelException() throws IOException, ModelException {
        Path modelDir = Paths.get("build/non-exist-model");
        String modelName = "mockModel";

        try (Model model = Model.newInstance(modelName)) {
            model.load(modelDir);
        }
    }
}
