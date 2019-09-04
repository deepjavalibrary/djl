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
package software.amazon.ai.examples.inference;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.imageio.ImageIO;
import org.apache.mxnet.zoo.ModelNotFoundException;
import org.apache.mxnet.zoo.ModelZoo;
import org.apache.mxnet.zoo.ZooModel;
import software.amazon.ai.Context;
import software.amazon.ai.examples.inference.util.AbstractExample;
import software.amazon.ai.examples.inference.util.Arguments;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.modality.cv.DetectedObject;
import software.amazon.ai.modality.cv.Images;
import software.amazon.ai.translate.TranslateException;

public final class SsdExample extends AbstractExample {

    public static void main(String[] args) {
        new SsdExample().runExample(args);
    }

    @Override
    public DetectedObject predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelNotFoundException, TranslateException {
        List<DetectedObject> predictResult = null;
        String modelName = arguments.getModelName();
        Path imageFile = arguments.getImageFile();
        BufferedImage img = Images.loadImageFromFile(imageFile);

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("size", "512");
        criteria.put("backbone", modelName);
        criteria.put("dataset", "voc");
        ZooModel<BufferedImage, List<DetectedObject>> model = ModelZoo.SSD.loadModel(criteria);

        // Following context is not not required, default context will be used by Predictor without
        // passing context to model.newPredictor(translator)
        // Change to a specific context if needed.
        Context context = Context.defaultContext();

        try (Predictor<BufferedImage, List<DetectedObject>> predictor =
                model.newPredictor(context)) {
            predictor.setMetrics(metrics); // Let predictor collect metrics

            for (int i = 0; i < iteration; ++i) {
                predictResult = predictor.predict(img);
                printProgress(iteration, i);
                collectMemoryInfo(metrics);
            }
        }
        drawBoundingBox(img, predictResult, arguments.getLogDir());

        model.close();
        return predictResult.get(0);
    }

    private void drawBoundingBox(
            BufferedImage img, List<DetectedObject> predictResult, String logDir)
            throws IOException {
        if (logDir == null) {
            return;
        }

        Path dir = Paths.get(logDir);
        Files.createDirectories(dir);

        Images.drawBoundingBox(img, predictResult);

        Path out = Paths.get(logDir, "ssd.jpg");
        ImageIO.write(img, "jpg", out.toFile());
    }
}
