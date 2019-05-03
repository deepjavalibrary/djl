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
package org.apache.mxnet.example;

import com.amazon.ai.Context;
import com.amazon.ai.example.util.AbstractExample;
import com.amazon.ai.example.util.LogUtils;
import com.amazon.ai.image.Images;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import com.sun.jna.Pointer;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;

public final class PredictApiExample extends AbstractExample {

    private static final Logger logger = LogUtils.getLogger(PredictApiExample.class);

    private PredictApiExample() {}

    public static void main(String[] args) {
        new PredictApiExample().runExample(args);
    }

    @Override
    public void predict(String modelDir, String modelName, BufferedImage img, int iteration)
            throws IOException {
        String symbolFile = modelDir + "/" + modelName + "-symbol.json";
        String paramFile = modelDir + "/" + modelName + "-0000.params";
        File synsetFile = new File(modelDir, "/synset.txt");

        byte[] symbol = Files.readAllBytes(Paths.get(symbolFile));
        String symbolJson = new String(symbol, StandardCharsets.UTF_8);
        byte[] buf = Files.readAllBytes(Paths.get(paramFile));
        ByteBuffer param = ByteBuffer.allocateDirect(buf.length);
        param.put(buf);
        param.rewind();
        String[] synset = MxModel.loadSynset(synsetFile);

        BufferedImage image = Images.reshapeImage(img, 224, 224);
        FloatBuffer data = Images.toFloatBuffer(image);

        Context context = Context.defaultContext();
        DataDesc dataDesc = new DataDesc(new Shape(1, 3, 224, 224), "data");

        long init = System.nanoTime();
        Pointer predictor = JnaUtils.createPredictor(symbolJson, param, context, dataDesc);
        long loadModel = System.nanoTime();
        logger.info(String.format("bind model  = %.3f ms.", (loadModel - init) / 1000000f));

        List<Long> inferenceTime = new ArrayList<>(iteration);
        for (int i = 0; i < iteration; ++i) {
            long begin = System.nanoTime();

            JnaUtils.setPredictorInput(predictor, "data", data);
            JnaUtils.predictorForward(predictor);
            FloatBuffer output = JnaUtils.getPredictorOutput(predictor);

            long inference = System.nanoTime();

            if (i == 0) {
                int idx = getResult(output);
                String className = synset[idx].trim();
                logger.info(String.format("Result: %s", className));
            }
            inferenceTime.add(inference - begin);
        }
        JnaUtils.freePredictor(predictor);

        Collections.sort(inferenceTime);

        float p50 = inferenceTime.get(iteration / 2) / 1000000f;
        float p90 = inferenceTime.get(iteration * 9 / 10) / 1000000f;

        logger.info(String.format("inference P50: %.3f ms, P90: %.3f ms", p50, p90));
    }

    public static int getResult(FloatBuffer data) {
        float bestAccuracy = 0.0f;
        int bestIdx = 0;

        for (int i = 0; i < data.limit(); i++) {
            if (data.get(i) > bestAccuracy) {
                bestAccuracy = data.get(i);
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}
