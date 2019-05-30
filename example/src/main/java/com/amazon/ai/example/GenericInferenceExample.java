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
package com.amazon.ai.example;

import com.amazon.ai.Model;
import com.amazon.ai.Translator;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.example.util.AbstractExample;
import com.amazon.ai.example.util.LogUtils;
import com.amazon.ai.image.Images;
import com.amazon.ai.inference.DetectedObject;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.metric.Metrics;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;

public final class GenericInferenceExample extends AbstractExample {

    private static final Logger logger = LogUtils.getLogger(GenericInferenceExample.class);

    private GenericInferenceExample() {}

    public static void main(String[] args) {
        new GenericInferenceExample().runExample(args);
    }

    @Override
    public void predict(File modelDir, String modelName, BufferedImage img, int iteration)
            throws IOException {
        Model model = Model.loadModel(modelDir, modelName);

        BufferedImage image = Images.reshapeImage(img, 224, 224);
        FloatBuffer data = Images.toFloatBuffer(image);

        GenericTranslator translator = new GenericTranslator(5, 224, 224);
        Metrics metrics = new Metrics();

        long init = System.nanoTime();
        try (Predictor<FloatBuffer, List<DetectedObject>> predictor =
                Predictor.newInstance(model, translator)) {

            predictor.setMetrics(metrics);

            long loadModel = System.nanoTime();
            logger.info(String.format("bind model  = %.3f ms.", (loadModel - init) / 1000000f));

            for (int i = 0; i < iteration; ++i) {
                List<DetectedObject> result = predictor.predict(data);
                printProgress(iteration, i, result.get(0).getClassName());
            }

            float p50 = metrics.percentile("Inference", 50).getValue() / 1000000f;
            float p90 = metrics.percentile("Inference", 90).getValue() / 1000000f;

            logger.info(String.format("inference P50: %.3f ms, P90: %.3f ms", p50, p90));
        }
    }

    private static final class GenericTranslator
            implements Translator<FloatBuffer, List<DetectedObject>> {

        private int topK;
        private DataDesc dataDesc;

        public GenericTranslator(int topK, int imageWidth, int imageHeight) {
            this.topK = topK;
            dataDesc = new DataDesc(new Shape(1, 3, imageWidth, imageHeight), "data");
        }

        @Override
        public NDList processInput(TranslatorContext ctx, FloatBuffer input) {
            NDArray array = ctx.getNDFactory().create(dataDesc);
            array.set(input);
            return new NDList(array);
        }

        @Override
        public List<DetectedObject> processOutput(TranslatorContext ctx, NDList list) {
            Model model = ctx.getModel();
            NDArray array = list.get(0);

            int length = array.getShape().head();
            length = Math.min(length, topK);
            List<DetectedObject> ret = new ArrayList<>(length);
            NDArray nd = array.at(0);
            NDArray sorted = nd.argsort(-1, false);
            NDArray top = sorted.slice(0, topK);

            float[] probabilities = nd.toFloatArray();
            float[] indices = top.toFloatArray();

            for (int i = 0; i < topK; ++i) {
                int index = (int) indices[i];
                String className = model.getSynset()[index];
                DetectedObject output = new DetectedObject(className, probabilities[index], null);
                ret.add(output);
            }
            return ret;
        }
    }
}
