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
package ai.djl.examples.inference.benchmark;

import ai.djl.examples.inference.benchmark.util.AbstractBenchmark;
import ai.djl.examples.inference.benchmark.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.listener.MemoryTrainingListener;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;

public final class Benchmark extends AbstractBenchmark<BufferedImage, Classifications> {

    BufferedImage img;
    Predictor<BufferedImage, Classifications> predictor;

    public static void main(String[] args) {
        if (new Benchmark().runBenchmark(args)) {
            System.exit(0); // NOPMD
        }
        System.exit(-1); // NOPMD
    }

    /** {@inheritDoc} */
    @Override
    protected void initialize(
            ZooModel<BufferedImage, Classifications> model, Arguments arguments, Metrics metrics)
            throws IOException {
        Path imageFile = arguments.getImageFile();
        img = BufferedImageUtils.fromFile(imageFile);
        predictor = model.newPredictor();
        predictor.setMetrics(metrics);
    }

    /** {@inheritDoc} */
    @Override
    protected CompletableFuture<Classifications> predict(
            ZooModel<BufferedImage, Classifications> model, Arguments arguments, Metrics metrics)
            throws TranslateException {

        Classifications result = predictor.predict(img);
        MemoryTrainingListener.collectMemoryInfo(metrics);
        return CompletableFuture.completedFuture(result);
    }

    /** {@inheritDoc} */
    @Override
    protected void clean() {
        predictor.close();
    }
}
