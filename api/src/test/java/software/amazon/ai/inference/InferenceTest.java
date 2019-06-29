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
package software.amazon.ai.inference;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import software.amazon.ai.Model;
import software.amazon.ai.TranslateException;
import software.amazon.ai.Translator;
import software.amazon.ai.TranslatorContext;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.modality.Classification;
import software.amazon.ai.modality.cv.BoundingBox;
import software.amazon.ai.modality.cv.DetectedObject;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.test.mock.EchoTranslator;
import software.amazon.ai.test.mock.MockImageTranslator;

public class InferenceTest {

    private BufferedImage image;

    @BeforeClass
    public void setup() throws IOException {
        Files.createDirectories(Paths.get("build/model"));
        image = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
    }

    @Test
    public void testObjectDection() throws IOException, TranslateException {
        Path modelDir = Paths.get("build/model");
        String modelName = "mockModel";

        Model model = Model.loadModel(modelDir, modelName);
        MockImageTranslator translator = new MockImageTranslator("cat");

        Metrics metrics = new Metrics();
        try (Predictor<BufferedImage, List<DetectedObject>> ssd =
                Predictor.newInstance(model, translator)) {
            ssd.setMetrics(metrics);
            List<DetectedObject> result = ssd.predict(image);
            DetectedObject detectedObject = result.get(0);
            Assert.assertEquals(detectedObject.getClassName(), "cat");

            BoundingBox box = detectedObject.getBoundingBox();
            Assert.assertEquals(Double.compare(box.getBounds().getHeight(), 1d), 0);
        }

        int p50 = metrics.percentile("Preprocess", 50).getValue().intValue();
        Assert.assertEquals(p50, 1);
        Assert.assertEquals(Double.compare(metrics.mean("Inference"), 3d), 0);
    }

    @Test
    public void testClassifier() throws IOException, TranslateException {
        Path modelDir = Paths.get("build/model");
        String modelName = "mockModel";

        Model model = Model.loadModel(modelDir, modelName);

        final String data = "cat";
        Translator<String, Classification> translator =
                new Translator<String, Classification>() {

                    @Override
                    public NDList processInput(TranslatorContext ctx, String input) {
                        return null;
                    }

                    @Override
                    public Classification processOutput(TranslatorContext ctx, NDList list) {
                        return new Classification(data, 0.9d);
                    }
                };
        Metrics metrics = new Metrics();

        try (Predictor<String, Classification> classifier =
                Predictor.newInstance(model, translator)) {
            classifier.setMetrics(metrics);
            Classification result = classifier.predict(data);
            Assert.assertEquals(result.getClassName(), "cat");
            Assert.assertEquals(Double.compare(result.getProbability(), 0.9d), 0);
        }
    }

    @Test(expectedExceptions = TranslateException.class)
    public void testTranslatException() throws TranslateException {
        EchoTranslator<String> translator = new EchoTranslator<>();
        translator.setInputException(new TranslateException("Some exception"));
        Predictor<String, String> predictor = Predictor.newInstance(null, translator);
        String result = predictor.predict("input");
        Assert.assertEquals(result, "input");
    }

    @Test(expectedExceptions = IOException.class)
    public void loadModelException() throws IOException {
        Path modelDir = Paths.get("build/non-exist-model");
        String modelName = "mockModel";

        Model.loadModel(modelDir, modelName);
    }
}
