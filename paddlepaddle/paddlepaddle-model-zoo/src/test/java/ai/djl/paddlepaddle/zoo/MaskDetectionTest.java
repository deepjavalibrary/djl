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

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
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

    private static Predictor<Image, Classifications> getClassifier()
            throws MalformedModelException, ModelNotFoundException, IOException {
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optTranslator(
                                ImageClassificationTranslator.builder()
                                        .addTransform(new Resize(128, 128))
                                        .addTransform(new ToTensor())
                                        .addTransform(
                                                new Normalize(
                                                        new float[] {0.5f, 0.5f, 0.5f},
                                                        new float[] {1.0f, 1.0f, 1.0f}))
                                        .addTransform(nd -> nd.flip(0)) // RGB -> GBR
                                        .build())
                        .optArtifactId("mask_classification")
                        .optFilter("flavor", "server")
                        .build();
        ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
        return model.newPredictor();
    }

    private static DetectedObjects detectFaces(Image img)
            throws ModelException, IOException, TranslateException {
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optArtifactId("face_detection")
                        .optTranslator(new FaceTranslator(0.5f, 0.7f))
                        .optFilter("flavor", "server")
                        .build();

        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria);
                Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            return predictor.predict(img);
        }
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

    static class FaceTranslator implements Translator<Image, DetectedObjects> {

        private float shrink;
        private float threshold;
        private List<String> className;

        FaceTranslator(float shrink, float threshold) {
            this.shrink = shrink;
            this.threshold = threshold;
            className = Arrays.asList("Not Face", "Face");
        }

        @Override
        public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
            NDArray result = list.singletonOrThrow();
            float[] probabilities = result.get(":,1").toFloatArray();
            List<String> names = new ArrayList<>();
            List<Double> prob = new ArrayList<>();
            List<BoundingBox> boxes = new ArrayList<>();
            for (int i = 0; i < probabilities.length; i++) {
                if (probabilities[i] >= threshold) {
                    float[] array = result.get(i).toFloatArray();
                    names.add(className.get((int) array[0]));
                    prob.add((double) probabilities[i]);
                    boxes.add(
                            new Rectangle(
                                    array[2], array[3], array[4] - array[2], array[5] - array[3]));
                }
            }
            return new DetectedObjects(names, prob, boxes);
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager());
            Shape shape = array.getShape();
            array =
                    NDImageUtils.resize(
                            array, (int) (shape.get(1) * shrink), (int) (shape.get(0) * shrink));
            array = array.transpose(2, 0, 1).flip(0); // HWC -> CHW
            NDArray mean =
                    ctx.getNDManager().create(new float[] {104f, 117f, 123f}, new Shape(3, 1, 1));
            array = array.sub(mean).mul(0.007843f); // normalization
            array = array.expandDims(0); // make batch dimension
            return new NDList(array);
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }
}
