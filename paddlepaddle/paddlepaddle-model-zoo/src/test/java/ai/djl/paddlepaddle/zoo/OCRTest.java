/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.List;
import org.testng.annotations.Test;

public class OCRTest {

    @Test
    public void testOCR() throws IOException, ModelException, TranslateException {
        String url = "https://resources.djl.ai/images/flight_ticket.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);
        List<DetectedObjects.DetectedObject> boxes = detectWords(img).items();

        // load Character model
        Predictor<Image, String> recognizer = getRecognizer();
        Predictor<Image, Classifications> rotator = getRotateClassifer();
        for (DetectedObjects.DetectedObject box : boxes) {
            Image subImg = getSubImage(img, box.getBoundingBox());
            if (subImg.getHeight() * 1.0 / subImg.getWidth() > 1.5) {
                subImg = rotateImg(subImg);
            }
            Classifications.Classification result = rotator.predict(subImg).best();
            if ("Rotate".equals(result.getClassName()) && result.getProbability() > 0.8) {
                subImg = rotateImg(subImg);
            }
            String name = recognizer.predict(subImg);
            System.out.println(name);
        }
    }

    private static DetectedObjects detectWords(Image img)
            throws ModelException, IOException, TranslateException {
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optArtifactId("ai.djl.paddlepaddle:word_detection")
                        .optFilter("flavor", "mobile")
                        .build();
        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
                Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            return predictor.predict(img);
        }
    }

    private static Predictor<Image, String> getRecognizer()
            throws MalformedModelException, ModelNotFoundException, IOException {
        Criteria<Image, String> criteria =
                Criteria.builder()
                        .setTypes(Image.class, String.class)
                        .optArtifactId("ai.djl.paddlepaddle:word_recognition")
                        .optFilter("flavor", "mobile")
                        .build();

        ZooModel<Image, String> model = criteria.loadModel();
        return model.newPredictor();
    }

    private static Predictor<Image, Classifications> getRotateClassifer()
            throws MalformedModelException, ModelNotFoundException, IOException {
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optArtifactId("ai.djl.paddlepaddle:word_rotation")
                        .optFilter("flavor", "mobile")
                        .build();
        ZooModel<Image, Classifications> model = criteria.loadModel();
        return model.newPredictor();
    }

    private static Image rotateImg(Image image) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray rotated = NDImageUtils.rotate90(image.toNDArray(manager), 1);
            return ImageFactory.getInstance().fromNDArray(rotated);
        }
    }

    private static Image getSubImage(Image img, BoundingBox box) {
        Rectangle rect = box.getBounds();
        double[] extended = extendRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight());
        int width = img.getWidth();
        int height = img.getHeight();
        int[] recovered = {
            (int) (extended[0] * width),
            (int) (extended[1] * height),
            (int) (extended[2] * width),
            (int) (extended[3] * height)
        };
        return img.getSubimage(recovered[0], recovered[1], recovered[2], recovered[3]);
    }

    private static double[] extendRect(double xmin, double ymin, double width, double height) {
        double centerx = xmin + width / 2;
        double centery = ymin + height / 2;
        if (width > height) {
            width += height * 2.0;
            height *= 3.0;
        } else {
            height += width * 2.0;
            width *= 3.0;
        }
        double newX = centerx - width / 2 < 0 ? 0 : centerx - width / 2;
        double newY = centery - height / 2 < 0 ? 0 : centery - height / 2;
        double newWidth = newX + width > 1 ? 1 - newX : width;
        double newHeight = newY + height > 1 ? 1 - newY : height;
        return new double[] {newX, newY, newWidth, newHeight};
    }
}
