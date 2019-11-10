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

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.modality.cv.util.NDImageUtils.Flag;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.zoo.cv.classification.Mlp;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ImageClassification {

    private static final Logger logger = LoggerFactory.getLogger(ImageClassification.class);

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Classifications classifications = new ImageClassification().predict();
        logger.info("{}", classifications);
    }

    public Classifications predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/0.png");
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);

        try (Model model = Model.newInstance()) {
            model.setBlock(new Mlp(28, 28));

            // Assume you have run TrainMnist.java example, and saved model in build/model folder.
            Path modelDir = Paths.get("build/model");
            model.load(modelDir, "mlp");

            Translator<BufferedImage, Classifications> translator = new MyTranslator();
            try (Predictor<BufferedImage, Classifications> predictor =
                    model.newPredictor(translator)) {
                return predictor.predict(img);
            }
        }
    }

    private static final class MyTranslator implements Translator<BufferedImage, Classifications> {

        private List<String> classes;

        public MyTranslator() {
            classes = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
        }

        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage input) {
            NDArray array = BufferedImageUtils.toNDArray(ctx.getNDManager(), input, Flag.COLOR);
            return new NDList(NDImageUtils.toTensor(array));
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            NDArray probabilities = list.singletonOrThrow().softmax(0);
            return new Classifications(classes, probabilities);
        }
    }
}
