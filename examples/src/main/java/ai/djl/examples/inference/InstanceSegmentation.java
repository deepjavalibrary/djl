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
package ai.djl.examples.inference;

import ai.djl.ModelException;
import ai.djl.examples.inference.util.AbstractInference;
import ai.djl.examples.inference.util.Arguments;
import ai.djl.examples.util.MemoryUtils;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.ImageVisualization;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.imageio.ImageIO;

public class InstanceSegmentation extends AbstractInference<DetectedObjects> {

    public static void main(String[] args) {
        new InstanceSegmentation().runExample(args);
    }

    /** {@inheritDoc} */
    @Override
    protected DetectedObjects predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelException, TranslateException {
        Path imageFile = arguments.getImageFile();
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("flavor", "v1b");
        criteria.put("backbone", "resnet18");
        criteria.put("dataset", "coco");
        ZooModel<BufferedImage, DetectedObjects> model =
                MxModelZoo.MASK_RCNN.loadModel(criteria, new ProgressBar());

        DetectedObjects result;
        try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
            predictor.setMetrics(metrics); // Let predictor collect metrics
            result = predictor.predict(img);

            MemoryUtils.collectMemoryInfo(metrics);
        }

        model.close();
        drawBoundingBox(img, result, arguments.getLogDir());
        return result;
    }

    private void drawBoundingBox(BufferedImage img, DetectedObjects predictResult, String logDir)
            throws IOException {
        if (logDir == null) {
            return;
        }

        Path dir = Paths.get(logDir);
        Files.createDirectories(dir);

        ImageVisualization.drawBoundingBoxes(img, predictResult);

        Path out = Paths.get(logDir, "imgSeg.jpg");
        ImageIO.write(img, "jpg", out.toFile());
    }
}
