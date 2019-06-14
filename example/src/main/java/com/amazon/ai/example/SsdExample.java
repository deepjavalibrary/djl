/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.amazon.ai.example;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.TranslateException;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.example.util.AbstractExample;
import com.amazon.ai.example.util.Arguments;
import com.amazon.ai.image.Images;
import com.amazon.ai.image.Rectangle;
import com.amazon.ai.inference.DetectedObject;
import com.amazon.ai.inference.ImageTranslator;
import com.amazon.ai.inference.ObjectDetector;
import com.amazon.ai.metric.Metrics;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;

public final class SsdExample extends AbstractExample {

    public static void main(String[] args) {
        new SsdExample().runExample(args);
    }

    @Override
    public DetectedObject predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, TranslateException {
        List<DetectedObject> predictResult = null;
        File modelDir = arguments.getModelDir();
        String modelName = arguments.getModelName();
        File imageFile = arguments.getImageFile();
        BufferedImage img = Images.loadImageFromFile(imageFile);

        Model model = Model.loadModel(modelDir, modelName);

        SsdTranslator translator = new SsdTranslator(0.2f, 512, 512);

        // Following context is not not required, default context will be used by Predictor without
        // passing context to Predictor.newInstance(model, translator)
        // Change to a specific context if needed.
        Context context = Context.defaultContext();

        try (ObjectDetector<BufferedImage, List<DetectedObject>> ssd =
                new ObjectDetector<>(model, translator, context)) {
            ssd.setMetrics(metrics); // Let predictor collect metrics

            for (int i = 0; i < iteration; ++i) {
                predictResult = ssd.detect(img);
                printProgress(iteration, i);
                collectMemoryInfo(metrics);
            }
        }
        drawBoundingBox(img, predictResult, arguments.getLogDir());
        return predictResult.get(0);
    }

    private void drawBoundingBox(
            BufferedImage img, List<DetectedObject> predictResult, String logDir)
            throws IOException {
        if (logDir == null) {
            return;
        }

        BufferedImage newImg = Images.reshapeImage(img, 512, 512);
        Graphics2D g = (Graphics2D) newImg.getGraphics();
        g.drawImage(img, 0, 0, 512, 512, null);
        int stroke = 2;
        g.setStroke(new BasicStroke(stroke));

        for (DetectedObject result : predictResult) {
            String className = result.getClassName();
            Rectangle rect = result.getBoundingBox().getBounds();
            g.setPaint(Color.WHITE);
            g.drawRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight());
            drawText(g, className, rect, stroke, 4);
        }
        g.dispose();

        File out = new File(logDir, "ssd.jpg");
        ImageIO.write(newImg, "jpg", out);
    }

    private void drawText(Graphics2D g, String text, Rectangle rect, int stroke, int padding) {
        FontMetrics metrics = g.getFontMetrics();
        int x = rect.getX() + stroke / 2;
        int y = rect.getY() + +stroke / 2;
        int width = metrics.stringWidth(text) + padding * 2 - stroke / 2;
        int height = metrics.getHeight() + metrics.getDescent();
        int ascent = metrics.getAscent();
        java.awt.Rectangle background = new java.awt.Rectangle(x, y, width, height);
        g.fill(background);
        g.setPaint(Color.BLACK);
        g.drawString(text, x + padding, y + ascent);
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
            BufferedImage image = Images.reshapeImage(input, imageWidth, imageHeight);
            return super.processInput(ctx, image);
        }

        @Override
        public List<DetectedObject> processOutput(TranslatorContext ctx, NDList list)
                throws TranslateException {
            Model model = ctx.getModel();
            NDArray array = list.get(0);

            List<DetectedObject> ret = new ArrayList<>();

            try {
                String[] synset = model.getArtifact("synset.txt", AbstractExample::loadSynset);
                NDArray nd = array.at(0);
                int length = nd.getShape().head();
                for (int i = 0; i < length; ++i) {
                    try (NDArray item = nd.at(i)) {
                        float[] values = item.toFloatArray();
                        int classId = (int) values[0];
                        float probability = values[1];
                        if (classId > 0 && probability > threshold) {
                            String className = synset[classId];

                            double x = values[2] * imageWidth;
                            double y = values[3] * imageHeight;
                            double w = values[4] * imageHeight - x;
                            double h = values[5] * imageHeight - y;

                            Rectangle rect = new Rectangle((int) x, (int) y, (int) w, (int) h);
                            ret.add(new DetectedObject(className, probability, rect));
                        }
                    }
                }
            } catch (IOException e) {
                throw new TranslateException(e);
            }

            return ret;
        }
    }
}
