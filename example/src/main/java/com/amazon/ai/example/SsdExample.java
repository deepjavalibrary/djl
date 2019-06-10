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
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public final class SsdExample extends AbstractExample {

    private SsdExample() {}

    public static void main(String[] args) {
        new SsdExample().runExample(args);
    }

    @Override
    public DetectedObject predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, TranslateException {
        DetectedObject predictResult = null;
        File modelDir = new File(arguments.getModelDir());
        String modelName = arguments.getModelName();
        String imageFile = arguments.getImageFile();
        BufferedImage img = Images.loadImageFromFile(new File(imageFile));

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
                List<DetectedObject> result = ssd.detect(img);
                predictResult = result.get(0);
                printProgress(iteration, i);
                collectMemoryInfo(metrics);
            }
        }
        return predictResult;
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
                int length = array.getShape().head();
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

                            Rectangle rect = new Rectangle(x, y, w, h);
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
