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
import com.amazon.ai.Transformer;
import com.amazon.ai.example.util.AbstractExample;
import com.amazon.ai.example.util.LogUtils;
import com.amazon.ai.image.Images;
import com.amazon.ai.inference.DetectedObject;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.mxnet.engine.MxModel;
import org.slf4j.Logger;

public final class GenericInferenceExample extends AbstractExample {

    private static final Logger logger = LogUtils.getLogger(GenericInferenceExample.class);

    private GenericInferenceExample() {}

    public static void main(String[] args) {
        new GenericInferenceExample().runExample(args);
    }

    @Override
    public void predict(String modelDir, String modelName, BufferedImage img, int iteration)
            throws IOException {
        String modelPathPrefix = modelDir + '/' + modelName;

        Model model = Model.loadModel(modelPathPrefix, 0);

        DataDesc dataDesc = new DataDesc(new Shape(1, 3, 224, 224), "data");
        ((MxModel) model).setDataNames(dataDesc);

        BufferedImage image = Images.reshapeImage(img, 224, 224);
        FloatBuffer data = Images.toFloatBuffer(image);

        GenericTransformer transformer = new GenericTransformer(5);

        long init = System.nanoTime();
        try (Predictor<FloatBuffer, List<DetectedObject>> predictor =
                Predictor.newInstance(model, transformer)) {
            long loadModel = System.nanoTime();
            logger.info(String.format("bind model  = %.3f ms.", (loadModel - init) / 1000000f));

            List<Long> inferenceTime = new ArrayList<>(iteration);
            for (int i = 0; i < iteration; ++i) {
                List<DetectedObject> result = predictor.predict(data);

                inferenceTime.add(transformer.getInferenceTime());

                if (i == 0) {
                    logger.info(String.format("Result: %s", result.get(0).getClassName()));
                }
            }

            Collections.sort(inferenceTime);

            float p50 = inferenceTime.get(iteration / 2) / 1000000f;
            float p90 = inferenceTime.get(iteration * 9 / 10) / 1000000f;

            logger.info(String.format("inference P50: %.3f ms, P90: %.3f ms", p50, p90));
        }
    }

    private static final class GenericTransformer
            implements Transformer<FloatBuffer, List<DetectedObject>> {

        private int topK;

        private long begin;
        private long end;

        public GenericTransformer(int topK) {
            this.topK = topK;
        }

        @Override
        public NDList processInput(Predictor<?, ?> predictor, FloatBuffer input) {
            Model model = predictor.getModel();
            NDArray array = predictor.create(model.describeInput()[0]);
            array.set(input);

            NDList list = new NDList(array);
            begin = System.nanoTime();

            return list;
        }

        @Override
        public List<DetectedObject> processOutput(Predictor<?, ?> predictor, NDList list) {
            for (NDArray array : list) {
                array.waitAll();
            }
            end = System.nanoTime();

            Model model = predictor.getModel();

            NDArray array = list.get(0);

            int length = array.getShape().head();
            length = Math.min(length, topK);
            List<DetectedObject> ret = new ArrayList<>(length);
            try (NDArray nd = array.at(0);
                    NDArray sorted = nd.argsort(-1, false);
                    NDArray top = sorted.slice(0, topK)) {

                float[] probabilities = nd.toFloatArray();
                float[] indices = top.toFloatArray();

                for (int i = 0; i < topK; ++i) {
                    int index = (int) indices[i];
                    String className = model.getSynset()[index];
                    DetectedObject output =
                            new DetectedObject(className, probabilities[index], null);
                    ret.add(output);
                }
            }
            return ret;
        }

        public long getInferenceTime() {
            return end - begin;
        }
    }
}
