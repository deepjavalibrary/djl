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
import com.amazon.ai.image.Images;
import com.amazon.ai.inference.DetectedObject;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.metric.Metrics;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class GenericInferenceExample extends AbstractExample {

    public static void main(String[] args) {
        new GenericInferenceExample().runExample(args);
    }

    @Override
    public DetectedObject predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, TranslateException {
        DetectedObject predictResult = null;
        Path modelDir = arguments.getModelDir();
        String modelName = arguments.getModelName();
        Path imageFile = arguments.getImageFile();
        BufferedImage img = Images.loadImageFromFile(imageFile);

        Model model = Model.loadModel(modelDir, modelName);

        GenericTranslator translator = new GenericTranslator(5, 224, 224);

        // Following context is not not required, default context will be used by Predictor without
        // passing context to Predictor.newInstance(model, translator)
        // Change to a specific context if needed.
        Context context = Context.defaultContext();

        try (Predictor<BufferedImage, List<DetectedObject>> predictor =
                Predictor.newInstance(model, translator, context)) {
            predictor.setMetrics(metrics); // Let predictor collect metrics

            for (int i = 0; i < iteration; ++i) {
                List<DetectedObject> result = predictor.predict(img);
                predictResult = result.get(0);
                printProgress(iteration, i);
                collectMemoryInfo(metrics);
            }
        }
        return predictResult;
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
            BufferedImage image = Images.resizeImage(input, imageWidth, imageHeight);
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
            int[] indices = top.toIntArray();

            String[] synset;
            try {
                synset = model.getArtifact("synset.txt", AbstractExample::loadSynset);
            } catch (IOException e) {
                throw new TranslateException(e);
            }
            for (int i = 0; i < topK; ++i) {
                int index = indices[i];
                String className = synset[index];
                DetectedObject output = new DetectedObject(className, probabilities[index], null);
                ret.add(output);
            }
            return ret;
        }
    }
}
