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
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.BaseImageTranslator;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MaskDetectionTest {

    @Test(enabled = false)
    public void testMaskDetection() throws IOException, ModelException, TranslateException {
        String url =
                "https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v1.5/demo/mask_detection/python/images/mask.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);

        List<DetectedObjects.DetectedObject> faces = detectFaces(img).items();
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
                        .optTranslator(
                                new FaceTranslator(
                                        FaceTranslator.builder()
                                                .addTransform(new Resize(256, 256))
                                                .addTransform(new ToTensor())))
                        .optFilter("flavor", "mobile")
                        .build();

        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria);
                Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            return predictor.predict(img);
        }
    }

    static class FaceTranslator extends BaseImageTranslator<DetectedObjects> {

        /**
         * Constructs an ImageTranslator with the provided builder.
         *
         * @param builder the data to build with
         */
        public FaceTranslator(Builder builder) {
            super(builder);
        }

        @Override
        public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws Exception {
            return null;
        }

        public static Builder builder() {
            return new Builder();
        }

        static class Builder extends BaseBuilder<Builder> {

            @Override
            protected Builder self() {
                return this;
            }
        }
    }
}
