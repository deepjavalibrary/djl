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
import ai.djl.modality.nlp.generate.ContrastiveSeqBatchScheduler;
import ai.djl.modality.nlp.generate.LMBlock;
import ai.djl.modality.nlp.generate.SearchConfig;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.util.Pair;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public final class DynamicSequenceScheduler {

    private DynamicSequenceScheduler() {}

    public static void main(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException {
        mainContrastivePt();
    }

    public static boolean mainContrastivePt()
            throws ModelNotFoundException, MalformedModelException, IOException {
        String[] modelUrls = {"https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2.pt.zip"};

        Pair<Block, List<Model>> result = LLMBlock.getLMBlock(modelUrls, "PyTorch", "GPT2");
        LMBlock lmBlock = (LMBlock) result.getKey();
        List<Model> models = result.getValue();

        boolean testResult = true;
        try (NDManager manager = NDManager.newBaseManager()) {

            SearchConfig config = new SearchConfig();
            config.setMaxSeqLength(30);
            config.setAlpha(0.6f);
            config.setK(5);
            config.setPadTokenId(220);

            ContrastiveSeqBatchScheduler scheduler =
                    new ContrastiveSeqBatchScheduler(lmBlock, config);

            // [r'DeepMind Company is',
            // r'Memories follow me left and right. I can']
            NDArray inputIds =
                    manager.create(
                            new long[][] {
                                {220, 220, 220, 220, 220, 220, 29744, 28478, 5834, 318},
                                {13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460}
                            });
            NDArray batchUids = manager.create(new long[] {0, 1});

            // Init input. Contain both initForward and seqBatcher merge
            scheduler.addRequest(inputIds, batchUids);

            // Increment forward
            scheduler.incrementForward(2);
            scheduler.inferenceCall();

            // Add more batch (longer)
            // [r"When your legs don't work like they used to before And I can't sweep you off",
            //  r"There's a time that I remember, when I did not know"]
            NDArray inputIds2 =
                    manager.create(
                            new long[][] {
                                {
                                    2215, 534, 7405, 836, 470, 670, 588, 484, 973, 284, 878, 843,
                                    314, 460, 470, 16085, 345, 572
                                },
                                {
                                    220, 220, 220, 220, 220, 1858, 338, 257, 640, 326, 314, 3505,
                                    11, 618, 314, 750, 407, 760
                                }
                            });
            NDArray batchUids2 = manager.create(new long[] {2, 3});
            scheduler.addRequest(inputIds2, batchUids2);
            scheduler.incrementForward(2);

            // Add more batch (shorter)
            // [r"A person gets sent back"]
            NDArray inputIds3 = manager.create(new long[][] {{32, 1048, 3011, 1908, 736}});
            NDArray batchUids3 = manager.create(new long[] {4});

            scheduler.addRequest(inputIds3, batchUids3);
            scheduler.incrementForward(20);

            // Collect partial results [1, 2, 3]
            Map<Long, NDArray> output1 = scheduler.collectResults();

            NDArray expected0 =
                    manager.create(
                            new long[] {
                                29744, 28478, 5834, 318, 257, 3298, 3554, 287, 11666, 4430, 290,
                                2769, 4673, 13, 775, 1975, 326, 9552, 460, 5854, 1096, 262, 835,
                                356, 9427, 351, 674, 3160, 11, 290
                            });
            NDArray expected1 =
                    manager.create(
                            new long[] {
                                13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460, 470, 4043,
                                284, 766, 644, 4325, 1306, 13, 198, 198, 14592, 50256
                            });
            NDArray expected2 =
                    manager.create(
                            new long[] {
                                2215, 534, 7405, 836, 470, 670, 588, 484, 973, 284, 878, 843, 314,
                                460, 470, 16085, 345, 572, 616, 3625, 13, 198, 198, 1, 1026, 338,
                                655, 257, 2300, 286
                            });
            NDArray expected3 =
                    manager.create(
                            new long[] {
                                1858, 338, 257, 640, 326, 314, 3505, 11, 618, 314, 750, 407, 760,
                                644, 284, 466, 351, 3589, 13, 314, 2936, 588, 314, 373, 1016, 284,
                                4656, 13, 314, 1807
                            });
            NDArray expected4 =
                    manager.create(
                            new long[] {
                                32, 1048, 3011, 1908, 736, 284, 3770, 329, 1204, 611, 484, 4589,
                                257, 4065, 326, 318, 30124, 416, 1918, 13, 198, 198, 464, 5617,
                                3078, 8879, 287, 1737, 326, 2585
                            });

            // Collect the rest of the results [0, 4]
            boolean emptyBatch = scheduler.incrementForward(config.getMaxSeqLength());
            Map<Long, NDArray> output2 = scheduler.collectResults();

            testResult &= output1.get(1L).equals(expected1);
            testResult &= output1.get(2L).equals(expected2);
            testResult &= output1.get(3L).equals(expected3);
            testResult &= output2.get(0L).equals(expected0);
            testResult &= output2.get(4L).equals(expected4);
            testResult &= emptyBatch;
        }
        models.forEach(Model::close);
        return testResult;
    }
}
