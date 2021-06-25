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
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MaskDetectionTest {

    @Test
    public void testMaskDetection() throws IOException, ModelException, TranslateException {
        String url =
                "https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v1.5/demo/mask_detection/python/images/mask.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);

        DetectedObjects boxes = detectFaces(img);
        List<DetectedObjects.DetectedObject> faces = boxes.items();
        Assert.assertEquals(faces.size(), 3);

        // Start classification
        Predictor<Image, Classifications> classifier = getClassifier();
        List<String> names = new ArrayList<>();
        List<Double> prob = new ArrayList<>();
        List<BoundingBox> rect = new ArrayList<>();
        for (DetectedObjects.DetectedObject face : faces) {
            Image subImg = getSubImage(img, face.getBoundingBox());
            Classifications classifications = classifier.predict(subImg);
            names.add(classifications.best().getClassName());
            prob.add(face.getProbability());
            rect.add(face.getBoundingBox());
        }
        saveBoundingBoxImage(img, new DetectedObjects(names, prob, rect));
    }

    private static Image getSubImage(Image img, BoundingBox box) {
        Rectangle rect = box.getBounds();
        int width = img.getWidth();
        int height = img.getHeight();
        int[] squareBox =
                extendSquare(
                        rect.getX() * width,
                        rect.getY() * height,
                        rect.getWidth() * width,
                        rect.getHeight() * height,
                        0.18);
        return img.getSubimage(squareBox[0], squareBox[1], squareBox[2], squareBox[2]);
    }

    private static int[] extendSquare(
            double xmin, double ymin, double width, double height, double percentage) {
        double centerx = xmin + width / 2;
        double centery = ymin + height / 2;
        double maxDist = Math.max(width / 2, height / 2) * (1 + percentage);
        return new int[] {
            (int) (centerx - maxDist), (int) (centery - maxDist), (int) (2 * maxDist)
        };
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("test.png");
        // OpenJDK can't save jpg with alpha channel
        newImage.save(Files.newOutputStream(imagePath), "png");
    }

    private static Predictor<Image, Classifications> getClassifier()
            throws MalformedModelException, ModelNotFoundException, IOException {
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optArtifactId("ai.djl.paddlepaddle:mask_classification")
                        .optFilter("flavor", "server")
                        .build();
        ZooModel<Image, Classifications> model = criteria.loadModel();
        return model.newPredictor();
    }

    private static DetectedObjects detectFaces(Image img)
            throws ModelException, IOException, TranslateException {
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optArtifactId("ai.djl.paddlepaddle:face_detection")
                        .optFilter("flavor", "server")
                        .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
                Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            return predictor.predict(img);
        }
    }
}
