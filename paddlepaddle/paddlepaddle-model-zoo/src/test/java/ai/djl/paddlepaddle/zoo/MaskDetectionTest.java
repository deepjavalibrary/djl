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
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MaskDetectionTest {

    @Test(enabled = false)
    public void testMaskDetection() throws IOException, ModelException, TranslateException {
        String url =
                "https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v1.5/demo/mask_detection/python/images/mask.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);

        DetectedObjects boxes = detectFaces(img);
        saveBoundingBoxImage(img, boxes);
        List<DetectedObjects.DetectedObject> faces = boxes.items();
        Assert.assertEquals(faces.size(), 3);

        Rectangle rect = faces.get(0).getBoundingBox().getBounds();
        int width = img.getWidth();
        int height = img.getHeight();
        Image faceImg =
                img.getSubimage(
                        (int) (rect.getX() * width),
                        (int) (rect.getY() * height),
                        (int) (rect.getWidth() * width),
                        (int) (rect.getHeight() * height));

        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optArtifactId("mask_classification")
                        .optFilter("flavor", "mobile")
                        .build();

        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Classifications classifications = predictor.predict(faceImg);
            Assert.assertEquals(classifications.best().getClassName(), "MASK");
        }
    }

    private static DetectedObjects detectFaces(Image img)
            throws ModelException, IOException, TranslateException {
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optArtifactId("face_detection")
                        .optTranslator(new FaceTranslator(0.5f))
                        .optFilter("flavor", "mobile")
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
        List<String> className;

        FaceTranslator(float shrink) {
            this.shrink = shrink;
            className = Arrays.asList("Not Face", "Face");
        }

        @Override
        public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws Exception {
            float[] array = list.singletonOrThrow().toFloatArray();
            List<String> names = Collections.singletonList(className.get((int) array[0]));
            List<Double> prob = Collections.singletonList((double) array[1]);
            List<BoundingBox> boxes = Collections.singletonList(new Rectangle(array[2], array[3], array[4] - array[2], array[5] - array[3]));
            return new DetectedObjects(names, prob, boxes);
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
            NDArray array = input.toNDArray(ctx.getNDManager());
            Shape shape = array.getShape();
            array = NDImageUtils.resize(array, (int) (shape.get(0) * shrink), (int) (shape.get(1) * shrink));
            array = array.transpose(2, 0, 1).flip(0); // HWC -> CHW + RGB -> BGR
            NDArray mean = ctx.getNDManager().create(new float[]{104f, 117f, 123f}, new Shape(3, 1, 1));
            array = array.sub(mean).mul(0.007843f); // normalization
            return new NDList(array);
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
}
