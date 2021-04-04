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

package ai.djl.examples.inference.face;

import ai.djl.ModelException;
import ai.djl.examples.inference.face.model.FaceDetectedObjects;
import ai.djl.examples.inference.face.model.Landmark;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Point;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An example of inference using a face detection model.
 *
 * <p>See this <a
 * href="https://github.com/awslabs/djl/blob/master/examples/docs/face_detection.md">doc</a> for
 * information about this example.
 */
public final class LightFaceDetection {
    private static final Logger logger = LoggerFactory.getLogger(LightFaceDetection.class);

    private LightFaceDetection() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        FaceDetectedObjects detection = LightFaceDetection.predict();
        logger.info("{}", detection);
    }

    public static FaceDetectedObjects predict()
            throws IOException, ModelException, TranslateException {
        double confThresh = 0.85f;
        double nmsThresh = 0.45f;
        double[] variance = new double[]{0.1f, 0.2f};
        int topK = 5000;
        int[][] scales = new int[][]{{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
        int[] steps = new int[]{8, 16, 32, 64};
        String facePath = "src/test/resources/largest_selfie.jpg";

        BufferedImage bufImg = ImageIO.read(new File(facePath));
        Image img = ImageFactory.getInstance().fromImage(bufImg);
        img.getWrappedImage();

        Criteria<Image, FaceDetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, FaceDetectedObjects.class)
                        .optModelUrls("https://djl-model.oss-cn-hongkong.aliyuncs.com/ultranet.zip")
                        // Load model from local file, e.g: "file:///Users/calvin/pytorch_models/retinaface/"
                        // .optModelUrls("file:///path/to/model_dir/")
                        .optModelName("ultranet") // specify model file prefix
                        .optTranslator(
                                new FaceDetectionTranslator(confThresh, nmsThresh, variance, topK, scales, steps))
                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch") // Use PyTorch engine
                        .build();

        try (ZooModel<Image, FaceDetectedObjects> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<Image, FaceDetectedObjects> predictor = model.newPredictor()) {
                FaceDetectedObjects detection = predictor.predict(img);
                drawLandmarks(bufImg, detection.getLandmarks());
                saveBoundingBoxImage(img, detection);
                return detection;
            }
        }
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("ultranet_detected.png");
        newImage.save(Files.newOutputStream(imagePath), "png");
        logger.info("Face detection result image has been saved in: {}", imagePath);
    }

    private static void drawLandmarks(BufferedImage image, List<Landmark> landmarks) {
        Graphics2D g = (Graphics2D) image.getGraphics();
        try {
            g.setColor(new Color(246, 96, 0));
            BasicStroke bStroke = new BasicStroke(4, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER);
            g.setStroke(bStroke);
            for (Landmark landmark : landmarks) {
                for (Point point : landmark.getPoints()) {
                    g.drawRect((int) point.getX(), (int) point.getY(), 2, 2);
                }
            }
        } finally {
            g.dispose();
        }
    }
}
