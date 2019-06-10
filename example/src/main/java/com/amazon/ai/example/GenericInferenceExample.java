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

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.TranslateException;
import com.amazon.ai.Translator;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.example.util.AbstractExample;
import com.amazon.ai.example.util.Arguments;
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
    public void predict(Arguments arguments, int iteration) throws IOException, TranslateException {
        File modelDir = new File(arguments.getModelDir());
        String modelName = arguments.getModelName();
        String imageFile = arguments.getImageFile();
        BufferedImage img = Images.loadImageFromFile(new File(imageFile));

        Model model = Model.loadModel(modelDir, modelName);

        GenericTranslator translator = new GenericTranslator(5, 224, 224);
        Metrics metrics = new Metrics();

        // Following context is not not required, default context will be used by Predictor without
        // passing context to Predictor.newInstance(model, translator)
        // Change to a specific context if needed.
        Context context = Context.defaultContext();

        long init = System.nanoTime();
        try (Predictor<BufferedImage, List<DetectedObject>> predictor =
                Predictor.newInstance(model, translator, context)) {

            predictor.setMetrics(metrics);

            long loadModel = System.nanoTime();
            logger.info(String.format("bind model  = %.3f ms.", (loadModel - init) / 1000000f));

            for (int i = 0; i < iteration; ++i) {
                List<DetectedObject> result = predictor.predict(img);
                printProgress(iteration, i, result.get(0).getClassName());
                collectMemoryInfo(metrics);
            }

            float p50 = metrics.percentile("Inference", 50).getValue().longValue() / 1000000f;
            float p90 = metrics.percentile("Inference", 90).getValue().longValue() / 1000000f;

            logger.info(String.format("inference P50: %.3f ms, P90: %.3f ms", p50, p90));

            dumpMemoryInfo(metrics, arguments.getLogDir());
        }
    }

    private static final class GenericTranslator
            implements Translator<BufferedImage, List<DetectedObject>> {

        private int topK;
        private DataDesc dataDesc;
        private int imageWidth;
        private int imageHeight;

        public GenericTranslator(int topK, int imageWidth, int imageHeight) {
            this.topK = topK;
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            dataDesc = new DataDesc(new Shape(1, 3, imageWidth, imageHeight), "data");
        }

        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage input) {
            BufferedImage image = Images.reshapeImage(input, imageWidth, imageHeight);
            FloatBuffer buffer = Images.toFloatBuffer(image);

            NDArray array = ctx.getNDFactory().create(dataDesc);
            array.set(buffer);
            return new NDList(array);
        }

        @Override
        public List<DetectedObject> processOutput(TranslatorContext ctx, NDList list)
                throws TranslateException {
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

            String[] synset;
            try {
                synset = model.getArtifact("synset.txt", AbstractExample::loadSynset);
            } catch (IOException e) {
                throw new TranslateException(e);
            }
            for (int i = 0; i < topK; ++i) {
                int index = (int) indices[i];
                String className = synset[index];
                DetectedObject output = new DetectedObject(className, probabilities[index], null);
                ret.add(output);
            }
            return ret;
        }
    }
}
