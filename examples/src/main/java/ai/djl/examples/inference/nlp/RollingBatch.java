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
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.modality.nlp.generate.ContrastiveSeqBatchScheduler;
import ai.djl.modality.nlp.generate.SearchConfig;
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
import java.util.Map;

public final class RollingBatch {

    private static final Logger logger = LoggerFactory.getLogger(ImageClassification.class);

    private RollingBatch() {}

    public static void main(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        String[] ret = seqBatchSchedulerWithPyTorchContrastive();
        logger.info("{}", ret[0]);
    }

    public static String[] seqBatchSchedulerWithPyTorchContrastive()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        String url = "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_pt.zip";

        Criteria<NDList, CausalLMOutput> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, CausalLMOutput.class)
                        .optModelUrls(url)
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new DeferredTranslatorFactory())
                        .build();

        String[] testResult = new String[5];

        try (ZooModel<NDList, CausalLMOutput> model = criteria.loadModel();
                Predictor<NDList, CausalLMOutput> predictor = model.newPredictor();
                NDManager manager = model.getNDManager().newSubManager();
                HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("gpt2")) {

            SearchConfig config = new SearchConfig();
            config.setMaxSeqLength(30);
            config.setAlpha(0.6f);
            config.setK(5);
            long padTokenId = 220;
            config.setPadTokenId(padTokenId);

            ContrastiveSeqBatchScheduler scheduler =
                    new ContrastiveSeqBatchScheduler(predictor, config);

            // Initial input
            String[] inputs1 = {"DeepMind Company is", "Memories follow me left and right. I can"};
            NDArray inputIds1 =
                    TextGeneration.encodeWithPadding(manager, tokenizer, inputs1, padTokenId);
            NDArray batchUids1 = manager.create(new long[] {0, 1});

            // Contains both initForward and seqBatcher merge
            scheduler.addRequest(inputIds1, batchUids1);

            // Increment forward
            scheduler.incrementForward(2);

            // Add more batch (longer)
            String[] inputs2 = {
                "When your legs don't work like they used to before And I can't sweep you" + " off",
                "There's a time that I remember, when I did not know"
            };
            NDArray inputIds2 =
                    TextGeneration.encodeWithPadding(manager, tokenizer, inputs2, padTokenId);
            NDArray batchUids2 = manager.create(new long[] {2, 3});
            scheduler.addRequest(inputIds2, batchUids2);
            scheduler.incrementForward(2);

            // Add more batch (shorter)
            String[] inputs3 = {"A person gets sent back"};
            NDArray inputIds3 =
                    TextGeneration.encodeWithPadding(manager, tokenizer, inputs3, padTokenId);
            NDArray batchUids3 = manager.create(new long[] {4});

            scheduler.addRequest(inputIds3, batchUids3);
            scheduler.incrementForward(config.getMaxSeqLength());

            // Collect result
            Map<Long, NDArray> output = scheduler.collectResults();
            testResult[0] = tokenizer.decode(output.get(0L).toLongArray());
            testResult[1] = tokenizer.decode(output.get(1L).toLongArray());
            testResult[2] = tokenizer.decode(output.get(2L).toLongArray());
            testResult[3] = tokenizer.decode(output.get(3L).toLongArray());
            testResult[4] = tokenizer.decode(output.get(4L).toLongArray());
        }
        return testResult;
    }
}
