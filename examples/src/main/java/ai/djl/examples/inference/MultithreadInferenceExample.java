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
package ai.djl.examples.inference;

import ai.djl.Device;
import ai.djl.examples.inference.util.AbstractInference;
import ai.djl.examples.inference.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classification;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MultithreadInferenceExample extends AbstractInference<Classification> {

    private static final Logger logger = LoggerFactory.getLogger(MultithreadInferenceExample.class);

    public static void main(String[] args) {
        new MultithreadInferenceExample().runExample(args);
    }

    @Override
    public Classification predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelNotFoundException {
        Path imageFile = arguments.getImageFile();
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);

        // Device is not required, default device will be used by Model if not provided.
        // Change to a specific device if needed.
        Device device = Device.defaultDevice();

        Map<String, String> criteria = arguments.getCriteria();
        if (criteria == null) {
            criteria = new ConcurrentHashMap<>();
            criteria.put("layers", "18");
            criteria.put("flavor", "v1");
        }
        ZooModel<BufferedImage, Classification> model =
                MxModelZoo.RESNET.loadModel(criteria, device);

        try {
            return predictImpl2(img, iteration, model, metrics);
        } catch (InterruptedException | ExecutionException e) {
            logger.error(e.getMessage());
        }
        return null;
    }

    public Classification predictImpl1(
            BufferedImage img,
            int iteration,
            ZooModel<BufferedImage, Classification> model,
            Metrics metrics)
            throws ExecutionException, InterruptedException {
        int numOfThreads = Runtime.getRuntime().availableProcessors();
        Predictor<BufferedImage, Classification> predictor = model.newPredictor();
        List<PredictCallable> callables =
                Collections.nCopies(
                        numOfThreads,
                        new PredictCallable(predictor, img, metrics, iteration / numOfThreads));
        ExecutorService executorService = Executors.newFixedThreadPool(numOfThreads);
        List<Future<Classification>> futures = executorService.invokeAll(callables);
        Classification classification = null;
        for (Future<Classification> future : futures) {
            classification = future.get();
        }
        return classification;
    }

    public Classification predictImpl2(
            BufferedImage img,
            int iteration,
            ZooModel<BufferedImage, Classification> model,
            Metrics metrics)
            throws ExecutionException, InterruptedException {
        int numOfThreads = Runtime.getRuntime().availableProcessors();
        List<PredictorCallable> callables =
                Collections.nCopies(
                        numOfThreads,
                        new PredictorCallable(model, img, metrics, iteration / numOfThreads));
        ExecutorService executorService = Executors.newFixedThreadPool(numOfThreads);
        List<Future<Classification>> futures = executorService.invokeAll(callables);
        Classification classification = null;
        for (Future<Classification> future : futures) {
            classification = future.get();
        }
        return classification;
    }

    static class PredictCallable implements Callable<Classification> {

        private Predictor<BufferedImage, Classification> predictor;
        private BufferedImage img;
        private Metrics metrics;
        private int iteration;

        public PredictCallable(
                Predictor<BufferedImage, Classification> predictor,
                BufferedImage img,
                Metrics metrics,
                int iteration) {
            this.predictor = predictor;
            this.img = img;
            this.metrics = metrics;
            this.iteration = iteration;
        }

        @Override
        public Classification call() throws TranslateException {
            predictor.setMetrics(metrics);
            Classification result = null;
            for (int i = 0; i < iteration; i++) {
                result = predictor.predict(img);
            }
            return result;
        }
    }

    static class PredictorCallable implements Callable<Classification> {

        private Predictor<BufferedImage, Classification> predictor;
        private BufferedImage img;
        private Metrics metrics;
        private int iteration;

        public PredictorCallable(
                ZooModel<BufferedImage, Classification> model,
                BufferedImage img,
                Metrics metrics,
                int iteration) {
            this.predictor = model.newPredictor();
            this.img = img;
            this.metrics = metrics;
            this.iteration = iteration;
        }

        @Override
        public Classification call() throws TranslateException {
            predictor.setMetrics(metrics);
            Classification result = null;
            for (int i = 0; i < iteration; i++) {
                result = predictor.predict(img);
            }
            return result;
        }
    }
}
