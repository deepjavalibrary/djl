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

import ai.djl.Device;
import ai.djl.examples.inference.util.AbstractInference;
import ai.djl.examples.inference.util.Arguments;
import ai.djl.examples.util.MemoryUtils;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.ImageVisualization;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.imageio.ImageIO;

public final class SsdExample extends AbstractInference<DetectedObjects> {

    public static void main(String[] args) {
        new SsdExample().runExample(args);
    }

    @Override
    public DetectedObjects predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelNotFoundException, TranslateException {
        Path imageFile = arguments.getImageFile();
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);

        // Device is not not required, default device will be used by Model if not provided.
        // Change to a specific device if needed.
        Device device = Device.defaultDevice();

        Map<String, String> criteria = arguments.getCriteria();
        if (criteria == null) {
            criteria = new ConcurrentHashMap<>();
            criteria.put("size", "512");
            criteria.put("backbone", "resnet50_v1");
            criteria.put("dataset", "voc");
        }
        ZooModel<BufferedImage, DetectedObjects> model = MxModelZoo.SSD.loadModel(criteria, device);

        DetectedObjects predictResult = null;
        try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
            predictor.setMetrics(metrics); // Let predictor collect metrics

            for (int i = 0; i < iteration; ++i) {
                predictResult = predictor.predict(img);

                progressBar.printProgress(i);
                MemoryUtils.collectMemoryInfo(metrics);
            }
        }
        drawBoundingBox(img, predictResult, arguments.getLogDir());

        model.close();
        return predictResult;
    }

    private void drawBoundingBox(BufferedImage img, DetectedObjects predictResult, String logDir)
            throws IOException {
        if (logDir == null) {
            return;
        }

        Path dir = Paths.get(logDir);
        Files.createDirectories(dir);

        ImageVisualization.drawBoundingBoxes(img, predictResult);

        Path out = Paths.get(logDir, "ssd.jpg");
        ImageIO.write(img, "jpg", out.toFile());
    }
}
