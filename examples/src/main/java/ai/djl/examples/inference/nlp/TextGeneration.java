/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.MalformedModelException;
import ai.djl.examples.inference.ImageClassification;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.modality.nlp.generate.SearchConfig;
import ai.djl.modality.nlp.generate.TextGenerator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.DeferredTranslatorFactory;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public final class TextGeneration {

    private static final Logger logger = LoggerFactory.getLogger(ImageClassification.class);

    private TextGeneration() {}

    public static void main(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        String ret = generateTextWithPyTorch();
        logger.info("{}", ret);
    }

    public static String generateTextWithPyTorch()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        SearchConfig config = new SearchConfig();
        config.setMaxSeqLength(60);

        String url = "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_pt.zip";

        Criteria<NDList, CausalLMOutput> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, CausalLMOutput.class)
                        .optModelUrls(url)
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new DeferredTranslatorFactory())
                        .build();
        String input = "DeepMind Company is";

        try (ZooModel<NDList, CausalLMOutput> model = criteria.loadModel();
                Predictor<NDList, CausalLMOutput> predictor = model.newPredictor();
                NDManager manager = model.getNDManager().newSubManager();
                HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("gpt2")) {

            TextGenerator generator = new TextGenerator(predictor, "greedy", config);

            Encoding encoding = tokenizer.encode(input);
            long[] inputIds = encoding.getIds();
            NDArray inputIdArray = manager.create(inputIds).expandDims(0);

            NDArray output = generator.greedySearch(inputIdArray);
            long[] outputIds = output.toLongArray();
            return tokenizer.decode(outputIds);
        }
    }
}
