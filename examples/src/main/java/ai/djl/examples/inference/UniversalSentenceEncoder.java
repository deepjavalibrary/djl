/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An example of inference using an universal sentence encoder model from TensorFlow Hub.
 *
 * <p>Refer to <a href="https://tfhub.dev/google/universal-sentence-encoder/4">Universal Sentence
 * Encoder</a> on TensorFlow Hub for more information.
 */
public final class UniversalSentenceEncoder {
    private static final Logger logger = LoggerFactory.getLogger(UniversalSentenceEncoder.class);

    private UniversalSentenceEncoder() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        List<String> inputs = new ArrayList<>();
        inputs.add("The quick brown fox jumps over the lazy dog.");
        inputs.add("I am a sentence for which I would like to get its embedding");

        List<float[]> embeddings = UniversalSentenceEncoder.predict(inputs);
        if (embeddings == null) {
            logger.info("This example only works for TensorFlow Engine");
        } else {
            for (int i = 0; i < inputs.size(); i++) {
                logger.info(
                        "Embedding for: "
                                + inputs.get(i)
                                + "\n"
                                + Arrays.toString(embeddings.get(i)));
            }
        }
    }

    public static List<float[]> predict(List<String> inputs)
            throws MalformedModelException, ModelNotFoundException, IOException,
                    TranslateException {
        if (!"TensorFlow".equals(Engine.getInstance().getEngineName())) {
            return null;
        }

        String modelUrl =
                "https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder/4.tar.gz";

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.TEXT_EMBEDDING)
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(modelUrl)
                        .optTranslator(new MyTranslator())
                        .optProgress(new ProgressBar())
                        .build();
        try (ZooModel<NDList, NDList> model = ModelZoo.loadModel(criteria);
                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDManager manager = NDManager.newBaseManager()) {
            NDList outputs =
                    predictor.predict(
                            new NDList(
                                    inputs.stream()
                                            .map(manager::create)
                                            .collect(Collectors.toList())));
            return outputs.stream().map(NDArray::toFloatArray).collect(Collectors.toList());
        }
    }

    private static final class MyTranslator implements Translator<NDList, NDList> {

        MyTranslator() {}

        @Override
        public NDList processInput(TranslatorContext ctx, NDList inputs) {
            // manually stack for faster batch inference
            return new NDList(NDArrays.stack(inputs));
        }

        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list) {
            NDList result = new NDList();
            long numOutputs = list.singletonOrThrow().getShape().get(0);
            for (int i = 0; i < numOutputs; i++) {
                result.add(list.singletonOrThrow().get(i));
            }
            return result;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }
}
