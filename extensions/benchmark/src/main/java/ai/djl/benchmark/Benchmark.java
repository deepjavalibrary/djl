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
package ai.djl.benchmark;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.listener.MemoryTrainingListener;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class runs single threaded benchmark. */
public final class Benchmark extends AbstractBenchmark {

    private static final Logger logger = LoggerFactory.getLogger(Benchmark.class);

    /**
     * Main entry point.
     *
     * @param args command line arguments
     */
    public static void main(String[] args) {
        String arch = System.getProperty("os.arch");
        if (!"x86_64".equals(arch) && !"amd64".equals(arch)) {
            logger.warn("{} is not supported.", arch);
            return;
        }
        List<String> list = Arrays.asList(args);
        boolean success;
        if (!list.isEmpty() && "ndlist-gen".equals(list.get(0))) {
            success = NDListGenerator.generate(Arrays.copyOfRange(args, 1, args.length));
        } else {
            boolean multithreading = list.contains("-t") || list.contains("--threads");
            configEngines(multithreading);
            if (multithreading) {
                success = new MultithreadedBenchmark().runBenchmark(args);
            } else {
                success = new Benchmark().runBenchmark(args);
            }
        }
        if (!success) {
            System.exit(-1); // NOPMD
        }
    }

    /** {@inheritDoc} */
    @Override
    public float[] predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelException, TranslateException {
        Device device = Engine.getEngine(arguments.getEngine()).defaultDevice();
        try (ZooModel<Void, float[]> model = loadModel(arguments, metrics, device)) {
            float[] predictResult = null;

            try (Predictor<Void, float[]> predictor = model.newPredictor()) {
                predictor.predict(null); // warmup

                predictor.setMetrics(metrics); // Let predictor collect metrics
                metrics.addMetric("start", System.currentTimeMillis(), "mills");
                for (int i = 0; i < iteration; ++i) {
                    predictResult = predictor.predict(null);

                    progressBar.update(i);
                    MemoryTrainingListener.collectMemoryInfo(metrics);
                }
                metrics.addMetric("end", System.currentTimeMillis(), "mills");
            }
            return predictResult;
        }
    }

    private static void configEngines(boolean multithreading) {
        if (multithreading) {
            if (System.getProperty("ai.djl.pytorch.num_interop_threads") == null) {
                System.setProperty("ai.djl.pytorch.num_interop_threads", "1");
            }
            if (System.getProperty("ai.djl.pytorch.num_threads") == null) {
                System.setProperty("ai.djl.pytorch.num_threads", "1");
            }
        }
        if (System.getProperty("ai.djl.tflite.disable_alternative") == null) {
            System.setProperty("ai.djl.tflite.disable_alternative", "true");
        }
        if (System.getProperty("ai.djl.dlr.disable_alternative") == null) {
            System.setProperty("ai.djl.dlr.disable_alternative", "true");
        }
        if (System.getProperty("ai.djl.paddlepaddle.disable_alternative") == null) {
            System.setProperty("ai.djl.paddlepaddle.disable_alternative", "true");
        }
        if (System.getProperty("ai.djl.onnx.disable_alternative") == null) {
            System.setProperty("ai.djl.onnx.disable_alternative", "true");
        }
        if (System.getProperty("ai.djl.tensorrt.disable_alternative") == null) {
            System.setProperty("ai.djl.tensorrt.disable_alternative", "true");
        }
    }
}
