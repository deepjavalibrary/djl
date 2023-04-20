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
import ai.djl.engine.Engine;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.GPTConfig;
import ai.djl.translate.LMAdapter;
import ai.djl.translate.LMSearch;
import ai.djl.translate.SearchConfig;

import java.io.IOException;
import java.nio.file.Paths;

public final class TestLMSearch {

    private TestLMSearch() {}

    public static void main(String[] args) {
        mainContrastivePt(args);
        mainGreedy(args);
        mainBeam(args);
        mainBeamOnnx(args);
    }

    public static void mainContrastivePt(String[] args) {
        //        String[] modelUrls = {
        //
        // "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_init_hidden.pt",
        //
        // "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_hidden.pt"
        //        };
        //        GPTConfig gptConfig = new GPTConfig(modelUrls);
        //        gptConfig.numAttentionHeads = 20;
        //        gptConfig.numLayers = 36;
        //        gptConfig.hiddenStateDim = 768;
        //        gptConfig.logitsDim = 50257;
        //        gptConfig.kvDim = 64;

        String[] modelUrls = {
            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_init_hidden.pt",
            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_hidden.pt"
        };
        GPTConfig gptConfig = new GPTConfig(modelUrls);

        try (NDManager manager = NDManager.newBaseManager()) {
            LMAdapter lmAdapter = Engine.getEngine("PyTorch").newLMAdapter("GPT2", gptConfig);

            LMSearch lmSearch;
            lmSearch = new LMSearch(lmAdapter);
            SearchConfig config = new SearchConfig();
            config.maxSeqLength = 60;
            config.alpha = 0.6f;
            config.k = 3;

            // [r'DeepMind Company is',
            // r'Memories follow me left and right. I can']
            NDArray inputIds =
                    manager.create(
                            new long[][] {
                                {220, 220, 220, 220, 220, 220, 29744, 28478, 5834, 318},
                                {13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460}
                            });
            config.padTokenId = 220;
            config.suffixPadding = false;

            int numBatch = (int) inputIds.getShape().get(0);
            int initSeqSize = (int) inputIds.getShape().get(1);
            NDArray attentionMask =
                    manager.ones(new Shape(1, inputIds.getShape().get(-1)), DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);

            boolean suffixPadding = true;
            long[][] attentionMaskSlice = new long[numBatch][2];
            for (int i = 0; i < numBatch; i++) {
                long[] aSequence = inputIds.get("{},:", i).toLongArray();
                int idx = 0;
                while (idx < initSeqSize) {
                    if (suffixPadding && aSequence[idx] == config.padTokenId
                            || !suffixPadding && aSequence[idx] != config.padTokenId) {
                        break;
                    }
                    idx++;
                }
                attentionMaskSlice[i][0] = suffixPadding ? idx : 0;
                attentionMaskSlice[i][1] = suffixPadding ? initSeqSize : idx;
                attentionMask.set(
                        new NDIndex(
                                "{},{}:{}",
                                i,
                                suffixPadding ? idx : 0,
                                suffixPadding ? initSeqSize : idx),
                        0);
            }

            NDArray output =
                    lmSearch.contrastiveSearch(manager, inputIds, attentionMaskSlice, config);
            System.out.println(output.toDebugString(1000, 10, 10, 100, true));

            printDecode(output);
        } catch (ModelNotFoundException | MalformedModelException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void mainGreedy(String[] args) {
        String[] modelUrls = {
            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_init_hidden.pt",
            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_hidden.pt"
        };
        GPTConfig gptConfig = new GPTConfig(modelUrls);

        try (NDManager manager = NDManager.newBaseManager()) {
            LMAdapter lmAdapter = Engine.getEngine("PyTorch").newLMAdapter("GPT2", gptConfig);

            LMSearch lmSearch;
            lmSearch = new LMSearch(lmAdapter);
            SearchConfig config = new SearchConfig();
            config.maxSeqLength = 60;

            // [r'DeepMind Company is',
            // r'Memories follow me left and right. I can']
            NDArray inputIds =
                    manager.create(
                            new long[][] {
                                {220, 220, 220, 220, 220, 220, 29744, 28478, 5834, 318},
                                {13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460}
                            });
            config.padTokenId = 220;
            config.suffixPadding = false;

            NDArray output = lmSearch.greedySearch(inputIds, config);
            System.out.println(output.toDebugString(1000, 10, 10, 100, true));

            printDecode(output);

        } catch (ModelNotFoundException | MalformedModelException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void mainBeam(String[] args) {
        String[] modelUrls = {
            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_init_hidden.pt",
            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_hidden.pt"
        };
        GPTConfig gptConfig = new GPTConfig(modelUrls);

        try (NDManager manager = NDManager.newBaseManager()) {
            LMAdapter lmAdapter = Engine.getEngine("PyTorch").newLMAdapter("GPT2", gptConfig);

            LMSearch lmSearch;
            lmSearch = new LMSearch(lmAdapter);
            SearchConfig config = new SearchConfig();
            config.maxSeqLength = 60;
            config.beam = 3;

            // [r'DeepMind Company is',
            // r'Memories follow me left and right. I can']
            NDArray inputIds =
                    manager.create(
                            new long[][] {
                                {50256, 50256, 50256, 50256, 50256, 50256, 29744, 28478, 5834, 318},
                                {13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460}
                            });
            config.padTokenId = 50256;
            config.suffixPadding = false;

            NDArray output = lmSearch.beamSearch(inputIds, config);
            System.out.println(output.toDebugString(1000, 10, 10, 100, true));

            printDecode(output);

        } catch (ModelNotFoundException | MalformedModelException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void mainBeamOnnx(String[] args) {
        String[] modelUrls = {
            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/gpt2_onnx/decoder_model_merged.onnx"
        };
        GPTConfig gptConfig = new GPTConfig(modelUrls);

        try (NDManager manager = NDManager.newBaseManager()) {
            LMAdapter lmAdapter = Engine.getEngine("OnnxRuntime").newLMAdapter("GPT2", gptConfig);

            LMSearch lmSearch;
            lmSearch = new LMSearch(lmAdapter);
            SearchConfig config = new SearchConfig();
            config.maxSeqLength = 60;
            config.beam = 3;

            // [r'DeepMind Company is',
            // r'Memories follow me left and right. I can']
            NDArray inputIds =
                    manager.create(
                            new long[][] {
                                {220, 220, 220, 220, 220, 220, 29744, 28478, 5834, 318},
                                {13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460}
                                //                                {220, 29744, 28478, 5834, 318}
                            });
            config.padTokenId = 220;
            config.suffixPadding = false;
            // The positionIds is not effective in onnx model traced from huggingface optimum.

            NDArray output = lmSearch.beamSearch(inputIds, config);
            System.out.println(output.toDebugString(1000, 10, 10, 100, true));

            printDecode(output);
        } catch (ModelNotFoundException | MalformedModelException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void printDecode(NDArray output) throws IOException {
        // Decoding
        String tokenizerJson =
                "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/gpt2_onnx/tokenizer.json";
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerJson));

        System.out.println('\n');
        for (int i = 0; i < output.getShape().get(0); i++) {
            System.out.println(i + ":");
            long[] aSequence = output.get("{},:", i).toLongArray();
            System.out.println(tokenizer.decode(aSequence));
        }
        System.out.println('\n');
    }
}
