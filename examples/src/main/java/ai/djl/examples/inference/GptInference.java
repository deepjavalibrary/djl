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
package ai.djl.examples.inference;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.generate.GPTConfig;
import ai.djl.modality.nlp.generate.LMBlock;
import ai.djl.modality.nlp.generate.LMSearch;
import ai.djl.modality.nlp.generate.SearchConfig;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

public final class GptInference {

    private static final Logger logger = LoggerFactory.getLogger(ImageClassification.class);

    private GptInference() {}

    public static void main(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        testOnnx();
        testPt();
    }

    private static void testPt()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        String[] modelUrls = {
            "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2.pt.zip",
        };
        Pair<Block, List<Model>> result = LLMBlock.getLMBlock(modelUrls, "PyTorch", "GPT2");
        LMBlock lmBlock = (LMBlock) result.getKey();
        // An adapter class lmBlock along with the lmBlock.forward call is inevitable, because, as
        // shown
        // in comments in L168-170, the searching code should be general rather than specific to a
        // certain model.

        SearchConfig config = new SearchConfig();
        config.setMaxSeqLength(60);

        String[] input = new String[] {"DeepMind Company is"};
        try (Model model = Model.newInstance("GPT2PtGreedy")) {
            // Change "greedy" to "contrastive", it will call greedy search
            model.setBlock(new LMSearch(lmBlock, "greedy", config));

            try (Predictor<String[], String> predictor =
                    model.newPredictor(new GPTTranslator()); ) {
                // According to the last code review meeting, the translator's pre/post process only
                // takes care of the tokenizer's encoding and decoding part. It's also why Zach
                // proposed
                // to make LMSearch inherit AbstractBlock, so that it will be wrapped in a Model and
                // utilizes the translator
                String output = predictor.predict(input);

                String expected =
                        "DeepMind Company is a global leader in the field of artificial"
                            + " intelligence and artificial intelligence. We are a leading provider"
                            + " of advanced AI solutions for the automotive industry, including the"
                            + " latest in advanced AI solutions for the automotive industry. We are"
                            + " also a leading provider of advanced AI solutions for the automotive"
                            + " industry, including the";

                logger.info("{}", expected.equals(output));
            }
        }
        result.getValue().forEach(Model::close);
    }

    private static class GPTTranslator implements NoBatchifyTranslator<String[], String> {

        HuggingFaceTokenizer tokenizer;

        public GPTTranslator() {
            tokenizer = HuggingFaceTokenizer.newInstance("gpt2");
        }

        /** {@inheritDoc} */
        @Override
        public String processOutput(TranslatorContext ctx, NDList list) {
            long[] output = list.singletonOrThrow().toLongArray();
            return tokenizer.decode(output);
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, String[] input) {
            Encoding encoding = tokenizer.encode(input);
            long[] inputIdsLong = encoding.getIds();
            NDArray inputIds = ctx.getNDManager().create(inputIdsLong);
            return new NDList(inputIds.expandDims(0));
        }
    }

    private static void testOnnx()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        String url = "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2.onnx.zip";
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(url)
                        .optEngine("OnnxRuntime")
                        .build();
        String input = "Large language model is";
        int maxLength = 5;
        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDManager manager = NDManager.newBaseManager("PyTorch");
                HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("gpt2")) {

            Encoding encoding = tokenizer.encode(input);
            long[] inputIds = encoding.getIds();
            long[] attentionMask = encoding.getAttentionMask();

            NDArray use = manager.create(new boolean[] {true});
            use.setName("use_cache_branch");
            NDArray notUse = manager.create(new boolean[] {false});
            notUse.setName("use_cache_branch");
            NDList pastKeyValues = initPastKeyValues(manager, 1);

            for (int i = 0; i < maxLength; ++i) {
                NDArray useCacheBranch;
                if (i == 0) {
                    useCacheBranch = notUse;
                } else {
                    useCacheBranch = notUse;
                }
                NDArray inputArray = manager.create(inputIds).expandDims(0);
                inputArray.setName("input_ids");
                NDArray attentionMaskArray = manager.create(attentionMask).expandDims(0);
                attentionMaskArray.setName("attention_mask");

                NDList list = new NDList(inputArray, attentionMaskArray, useCacheBranch);
                list.addAll(pastKeyValues);
                NDList output = predictor.predict(list);
                // The list input here is specific to a certain model like onnx here,
                // which renders the searching algorithm work only specifically. Thus,
                // an adapter is needed here to make the searching code work for any model.

                NDArray logits = output.get(0);
                NDArray result = logits.get(new NDIndex(":,-1,:"));
                long nextToken = result.argMax().getLong();

                pastKeyValues = output.subNDList(1);
                int numLayer = pastKeyValues.size() / 2;
                for (int j = 0; j < numLayer; ++j) {
                    int index = j * 2;
                    pastKeyValues.get(index).setName("past_key_values." + j + ".key");
                    pastKeyValues.get(index + 1).setName("past_key_values." + j + ".value");
                }

                inputIds = expend(inputIds, nextToken);
                attentionMask = expend(attentionMask, 1);
            }

            logger.info(tokenizer.decode(inputIds));
        }
    }

    static long[] expend(long[] array, long item) {
        long[] ret = new long[array.length + 1];
        System.arraycopy(array, 0, ret, 0, array.length);
        ret[array.length] = item;
        return ret;
    }

    static NDList initPastKeyValues(NDManager manager, int numBatch) {
        GPTConfig config = new GPTConfig();
        long kvDim = config.getKvDim();
        int numAttentionHeads = config.getNumAttentionHeads();
        int numLayers = config.getNumLayers();

        NDList list = new NDList(2 * numLayers);
        for (int i = 0; i < numLayers; ++i) {
            NDArray key = manager.zeros(new Shape(numBatch, numAttentionHeads, 1, kvDim));
            key.setName("past_key_values." + i + ".key");
            NDArray value = manager.zeros(new Shape(numBatch, numAttentionHeads, 1, kvDim));
            value.setName("past_key_values." + i + ".value");
            list.add(key);
            list.add(value);
        }
        return list;
    }
}
