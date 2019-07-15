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
package software.amazon.ai.examples;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.TranslateException;
import software.amazon.ai.TranslatorContext;
import software.amazon.ai.examples.util.AbstractExample;
import software.amazon.ai.examples.util.Arguments;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.modality.cv.DetectedObject;
import software.amazon.ai.modality.cv.ImageTranslator;
import software.amazon.ai.modality.cv.Images;
import software.amazon.ai.modality.cv.Rectangle;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;

public final class SsdExample extends AbstractExample {

    public static void main(String[] args) {
        new SsdExample().runExample(args);
    }

    @Override
    public DetectedObject predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, TranslateException {
        List<DetectedObject> predictResult = null;
        Path modelDir = arguments.getModelDir();
        String modelName = arguments.getModelName();
        Path imageFile = arguments.getImageFile();
        BufferedImage img = Images.loadImageFromFile(imageFile);

        Model model = Model.loadModel(modelDir, modelName);

        SsdTranslator translator = new SsdTranslator(0.2f, 512, 512);

        // Following context is not not required, default context will be used by Predictor without
        // passing context to Predictor.newInstance(model, translator)
        // Change to a specific context if needed.
        Context context = Context.defaultContext();

        try (Predictor<BufferedImage, List<DetectedObject>> ssd =
                Predictor.newInstance(model, translator, context)) {
            ssd.setMetrics(metrics); // Let predictor collect metrics

            for (int i = 0; i < iteration; ++i) {
                predictResult = ssd.predict(img);
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

    private static final class SsdTranslator extends ImageTranslator<List<DetectedObject>> {

        private float threshold;
        private int imageWidth;
        private int imageHeight;

        public SsdTranslator(float threshold, int imageWidth, int imageHeight) {
            this.threshold = threshold;
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage input) {
            BufferedImage image = Images.resizeImage(input, imageWidth, imageHeight);
            return super.processInput(ctx, image);
        }

        @Override
        public List<DetectedObject> processOutput(TranslatorContext ctx, NDList list)
                throws IOException {
            Model model = ctx.getModel();
            NDArray array = list.get(0);

            List<DetectedObject> ret = new ArrayList<>();

            String[] synset = model.getArtifact("synset.txt", AbstractExample::loadSynset);
            NDArray nd = array.get(0);
            long length = nd.getShape().head();
            for (int i = 0; i < length; ++i) {
                try (NDArray item = nd.get(i)) {
                    float[] values = item.toFloatArray();
                    int classId = (int) values[0];
                    float probability = values[1];
                    if (classId > 0 && probability > threshold) {
                        if (classId >= synset.length) {
                            throw new AssertionError("Unexpected index: " + classId);
                        }
                        String className = synset[classId];

                        double x = values[2];
                        double y = values[3];
                        double w = values[4] - x;
                        double h = values[5] - y;

                        Rectangle rect = new Rectangle(x, y, w, h);
                        ret.add(new DetectedObject(className, probability, rect));
                    }
                }
            }

            return ret;
        }
    }
}
