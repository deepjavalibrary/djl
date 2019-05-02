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

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.Transformer;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.image.Images;
import com.amazon.ai.inference.DetectedObject;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import java.awt.image.BufferedImage;
import java.io.File;
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
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.jna.JnaUtils;

public final class JouleInferenceExample {

    private JouleInferenceExample() {}

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

            BufferedImage img = Images.loadImageFromFile(new File(imageFile));
            BufferedImage image = Images.reshapeImage(img, 224, 224);
            FloatBuffer data = Images.toFloatBuffer(image);

            long init = System.nanoTime();
            System.out.println("Loading native library: " + JnaUtils.getVersion());
            long loaded = System.nanoTime();
            System.out.printf("loadlibrary = %.3f ms.%n", (loaded - init) / 1000000f);

            while (!duration.isNegative()) {
                long begin = System.currentTimeMillis();
                predict(modelDir, modelName, data, iteration);
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

    private static void predict(String modelDir, String modelName, FloatBuffer data, int iteration)
            throws IOException {
        Context context = Context.defaultContext();

        String modelPathPrefix = modelDir + '/' + modelName;

        Model model = Model.loadModel(modelPathPrefix, 0);

        DataDesc dataDesc = new DataDesc(new Shape(1, 3, 224, 224), "data");
        ((MxModel) model).setDataNames(dataDesc);

        SampleTransformer transformer = new SampleTransformer(model, 5);

        long init = System.nanoTime();
        try (Predictor<FloatBuffer, List<DetectedObject>> predictor =
                Predictor.newInstance(model, transformer)) {
            long loadModel = System.nanoTime();
            System.out.printf("bind model  = %.3f ms.%n", (loadModel - init) / 1000000f);

            List<Long> inferenceTime = new ArrayList<>(iteration);
            for (int i = 0; i < iteration; ++i) {
                List<DetectedObject> result = predictor.predict(data);

                inferenceTime.add(transformer.getInferenceTime());

                if (i == 0) {
                    System.out.printf("Result: %s%n", result.get(0).getClassName());
                }
            }

            Collections.sort(inferenceTime);

            float p50 = inferenceTime.get(iteration / 2) / 1000000f;
            float p90 = inferenceTime.get(iteration * 9 / 10) / 1000000f;

            System.out.printf("inference P50: %.3f ms, P90: %.3f ms%n", p50, p90);
        }
    }

    private static final class SampleTransformer
            implements Transformer<FloatBuffer, List<DetectedObject>> {

        private Model model;
        private int topK;
        private NDFactory factory;

        private long begin;
        private long end;

        public SampleTransformer(Model model, int topK) {
            this.model = model;
            this.topK = topK;
            factory = Engine.getInstance().getNDFactory();
        }

        @Override
        public NDList processInput(FloatBuffer input) {
            NDArray array = factory.create(model.describeInput()[0]);
            array.set(input);

            NDList list = new NDList(array);
            begin = System.nanoTime();

            return list;
        }

        @Override
        public List<DetectedObject> processOutput(NDList list) {
            end = System.nanoTime();

            NDArray array = list.get(0);

            int length = array.getShape().head();
            length = Math.min(length, topK);
            List<DetectedObject> ret = new ArrayList<>(length);
            try (NDArray nd = array.at(0)) {
                NDArray sorted = nd.argsort(-1, false);
                NDArray top = sorted.slice(0, topK);

                float[] probabilities = nd.toFloatArray();
                float[] indices = top.toFloatArray();

                sorted.close();
                top.close();

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
