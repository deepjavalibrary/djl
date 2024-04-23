/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.ModelException;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;

/** An example of inference using Bert Text Embedding. */
public final class BertTextEmbedding {

    private static final Logger logger = LoggerFactory.getLogger(BertTextEmbedding.class);

    private BertTextEmbedding() {}

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        float[][] output = BertTextEmbedding.predict();
        logger.info("Output: {}", Arrays.deepToString(output));
    }

    public static float[][] predict() throws IOException, TranslateException, ModelException {
        String[] input = {"What is Deep Learning?", "The movie got Oscar this year"};
        Criteria<String[], float[][]> criteria =
                Criteria.builder()
                        .optModelUrls(
                                "https://alpha-djl-demos.s3.amazonaws.com/model/examples/bge-base-en-v1.5.zip")
                        .setTypes(String[].class, float[][].class)
                        .optEngine("Rust")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<String[], float[][]> model = criteria.loadModel()) {
            try (Predictor<String[], float[][]> predictor = model.newPredictor()) {
                return predictor.predict(input);
            }
        }
    }
}
