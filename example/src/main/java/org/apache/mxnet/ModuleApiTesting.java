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

package org.apache.mxnet;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.mxnet.image.Image;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.model.DataDesc;
import org.apache.mxnet.model.Module;
import org.apache.mxnet.model.MxModel;
import org.apache.mxnet.model.NdArray;
import org.apache.mxnet.model.ResourceAllocator;
import org.apache.mxnet.model.Shape;
import org.apache.mxnet.types.DataType;

@SuppressWarnings("PMD.SystemPrintln")
public final class ModuleApiTesting {

    private ModuleApiTesting() {}

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

            System.out.println("ModuleApiTesting: iteration: " + iteration);

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
                predict(modelDir, modelName, data, dataDesc, iteration);
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
            String modelDir, String modelName, FloatBuffer data, DataDesc dataDesc, int iteration)
            throws IOException {
        Context context = Context.getDefaultContext();

        String modelPathPrefix = modelDir + '/' + modelName;

        List<DataDesc> dataDescs = Collections.singletonList(dataDesc);
        Shape inputShape = dataDesc.getShape();

        try (ResourceAllocator alloc = new ResourceAllocator()) {
            MxModel model = MxModel.loadSavedModel(alloc, modelPathPrefix, 0);

            long init = System.nanoTime();
            Module.Builder builder = new Module.Builder(context, model, dataDescs, false);
            Module module = builder.build(alloc);
            long loadModel = System.nanoTime();
            System.out.printf("bind model  = %.3f ms.%n", (loadModel - init) / 1000000f);

            List<Long> inferenceTime = new ArrayList<>(iteration);
            for (int i = 0; i < iteration; ++i) {
                long begin = System.nanoTime();

                try (ResourceAllocator alloc1 = new ResourceAllocator()) {
                    NdArray ndArray = new NdArray(alloc1, context, inputShape);
                    ndArray.set(data);

                    NdArray[] input = new NdArray[] {ndArray};

                    module.forward(input);

                    NdArray[] ret = module.getOutputs();
                    for (NdArray nd : ret) {
                        nd.waitAll();
                    }

                    long inference = System.nanoTime();
                    inferenceTime.add(inference - begin);

                    NdArray sorted = ret[0].argsort(-1, false);
                    NdArray top = sorted.slice(0, 1);

                    float[] indices = top.toFloatArray();
                    String className = model.getSynset()[(int) indices[0]];

                    if (i == 0) {
                        System.out.printf("Result: %s%n", className);
                    }
                }
            }

            Collections.sort(inferenceTime);

            float p50 = inferenceTime.get(iteration / 2) / 1000000f;
            float p90 = inferenceTime.get(iteration * 9 / 10) / 1000000f;

            System.out.printf("inference P50: %.3f ms, P90: %.3f ms%n", p50, p90);
        }
    }
}
