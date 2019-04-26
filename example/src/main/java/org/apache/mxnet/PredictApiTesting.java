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
package org.apache.mxnet;

import com.sun.jna.Pointer;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.mxnet.engine.DataDesc;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.engine.Shape;
import org.apache.mxnet.image.Image;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.types.DataType;

@SuppressWarnings("PMD.SystemPrintln")
public final class PredictApiTesting {

    private PredictApiTesting() {}

    public static void main(String[] args) {
        Options options = Arguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = new Arguments(cmd);

            String modelDir = arguments.getModelDir();
            String modelName = arguments.getModelName();
            String imageFile = arguments.getImageFile();
            Duration duration = Duration.ofMinutes(arguments.getDuration());
            int iteration = arguments.getIteration();

            System.out.println("PredictApiTesting: iteration: " + iteration);

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

            Shape inputShape = new Shape(1, 3, 224, 224);
            DataDesc dataDesc = new DataDesc("data", inputShape, DataType.FLOAT32, "NCHW");

            BufferedImage img = Image.loadImageFromFile(imageFile);
            BufferedImage image = Image.reshapeImage(img, 224, 224);
            FloatBuffer data = Image.toDirectBuffer(image);

            long init = System.nanoTime();
            System.out.println("Loading native library: " + JnaUtils.getVersion());
            long loaded = System.nanoTime();
            System.out.printf("loadlibrary = %.3f ms.%n", (loaded - init) / 1000000f);

            while (!duration.isNegative()) {
                long begin = System.currentTimeMillis();
                predict(symbolJson, param, data, dataDesc, synset, iteration);
                long delta = System.currentTimeMillis() - begin;
                duration = duration.minus(Duration.ofMillis(delta));
            }
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            t.printStackTrace(); // NOPMD
        }
    }

    private static void predict(
            String symbolJson,
            ByteBuffer param,
            FloatBuffer data,
            DataDesc dataDesc,
            String[] synset,
            int iteration) {
        Context context = Context.getDefaultContext();

        long init = System.nanoTime();
        Pointer predictor = JnaUtils.createPredictor(symbolJson, param, context, dataDesc);
        long loadModel = System.nanoTime();

        System.out.printf("bind model  = %.3f ms.%n", (loadModel - init) / 1000000f);

        List<Long> inferenceTime = new ArrayList<>(iteration);
        for (int i = 0; i < iteration; ++i) {
            long begin = System.nanoTime();

            JnaUtils.setPredictorInput(predictor, "data", data);
            JnaUtils.predictorForward(predictor);
            FloatBuffer output = JnaUtils.getPredictorOutput(predictor);

            long inference = System.nanoTime();

            if (i == 0) {
                int idx = getResult(output);
                System.out.printf("Result: %s%n", synset[idx].trim());
                // long postprocess = System.nanoTime();
            }
            inferenceTime.add(inference - begin);
        }
        JnaUtils.freePredictor(predictor);

        Collections.sort(inferenceTime);

        float p50 = inferenceTime.get(iteration / 2) / 1000000f;
        float p90 = inferenceTime.get(iteration * 9 / 10) / 1000000f;

        System.out.printf("inference P50: %.3f ms, P90: %.3f ms%n", p50, p90);
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
