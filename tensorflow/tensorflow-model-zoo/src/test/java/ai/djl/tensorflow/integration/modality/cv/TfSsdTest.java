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
package ai.djl.tensorflow.integration.modality.cv;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TfSsdTest {

    @Test
    public void testTfSSD() throws IOException, ModelException, TranslateException {
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optArtifactId("ssd")
                        .optFilter("backbone", "mobilenet_v2")
                        .optEngine("TensorFlow")
                        .optProgress(new ProgressBar())
                        .build();

        Path file = Paths.get("../../examples/src/test/resources/dog_bike_car.jpg");
        Image img = ImageFactory.getInstance().fromFile(file);
        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
                Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            Assert.assertEquals(model.describeInput().get(0).getValue(), new Shape(-1, -1, -1, 3));
            for (Pair<String, Shape> pair : model.describeOutput()) {
                if (pair.getKey().contains("label")) {
                    Assert.assertEquals(pair.getValue(), new Shape(-1, 1));
                } else if (pair.getKey().contains("box")) {
                    Assert.assertEquals(pair.getValue(), new Shape(-1, 4));
                } else if (pair.getKey().contains("score")) {
                    Assert.assertEquals(pair.getValue(), new Shape(-1, 1));
                } else {
                    throw new IllegalStateException("Unexpected output name:" + pair.getKey());
                }
            }

            DetectedObjects result = predictor.predict(img);
            List<String> classes =
                    result.items()
                            .stream()
                            .map(Classifications.Classification::getClassName)
                            .collect(Collectors.toList());
            Assert.assertTrue(classes.contains("Dog"));
            Assert.assertTrue(classes.contains("Bicycle"));
            Assert.assertTrue(classes.contains("Car"));
            saveBoundingBoxImage(img, result);
        }
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("detected-dog_bike_car.png");
        // OpenJDK can't save jpg with alpha channel
        newImage.save(Files.newOutputStream(imagePath), "png");
    }
}
