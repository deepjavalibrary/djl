/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference.nlp;

import ai.djl.ModelException;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.EmbeddingOutput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public final class SparseRetrieval {

    private static final Logger logger = LoggerFactory.getLogger(SparseRetrieval.class);

    private SparseRetrieval() {}

    public static void main(String[] args) throws ModelException, IOException, TranslateException {
        List<EmbeddingOutput> ret = predict();
        for (EmbeddingOutput result : ret) {
            logger.info("text embedding: {}", JsonUtils.GSON_PRETTY.toJson(result));
        }
    }

    public static List<EmbeddingOutput> predict()
            throws ModelException, IOException, TranslateException {
        Criteria<String, EmbeddingOutput> criteria =
                Criteria.builder()
                        .setTypes(String.class, EmbeddingOutput.class)
                        .optModelUrls("djl://ai.djl.huggingface.pytorch/BAAI/bge-m3")
                        .optEngine("PyTorch")
                        .optArgument("sparse", true) // use SparseRetrievalTranslator
                        .optArgument("returnDenseEmbedding", false) // only returns sparse embedding
                        .optArgument("sparseLinear", "sparse_linear.safetensors")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .optProgress(new ProgressBar())
                        .build();
        List<String> inputs =
                Arrays.asList("This is an example sentence", "What is sparse retrieval?");
        try (ZooModel<String, EmbeddingOutput> model = criteria.loadModel();
                Predictor<String, EmbeddingOutput> predictor = model.newPredictor()) {
            return predictor.batchPredict(inputs);
        }
    }
}
