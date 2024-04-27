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
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.modality.nlp.generate.SearchConfig;
import ai.djl.modality.nlp.generate.TextGenerator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.DeferredTranslatorFactory;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public final class TextGeneration {

    private static final Logger logger = LoggerFactory.getLogger(TextGeneration.class);

    private TextGeneration() {}

    public static void main(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        String ret1 = generateTextWithPyTorchGreedy();
        logger.info("{}", ret1);
        String[] ret2 = generateTextWithPyTorchContrastive();
        logger.info("{}", ret2[0]);
        String[] ret3 = generateTextWithPyTorchBeam();
        logger.info("{}", ret3[0]);
    }

    public static String generateTextWithPyTorchGreedy()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        SearchConfig config = new SearchConfig();
        config.setMaxSeqLength(60);

        // You can use src/main/python/trace_gpt2.py to trace gpt2 model
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

            NDArray output = generator.generate(inputIdArray);
            long[] outputIds = output.toLongArray();
            return tokenizer.decode(outputIds);
        }
    }

    public static String[] generateTextWithPyTorchContrastive()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        SearchConfig config = new SearchConfig();
        config.setMaxSeqLength(60);
        long padTokenId = 220;
        config.setPadTokenId(padTokenId);

        String url = "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_pt.zip";

        Criteria<NDList, CausalLMOutput> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, CausalLMOutput.class)
                        .optModelUrls(url)
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new DeferredTranslatorFactory())
                        .build();
        String[] inputs = {"DeepMind Company is", "Memories follow me left and right. I can"};

        try (ZooModel<NDList, CausalLMOutput> model = criteria.loadModel();
                Predictor<NDList, CausalLMOutput> predictor = model.newPredictor();
                NDManager manager = model.getNDManager().newSubManager();
                HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("gpt2")) {

            TextGenerator generator = new TextGenerator(predictor, "contrastive", config);
            NDArray inputIdArray = encodeWithPadding(manager, tokenizer, inputs, padTokenId);

            NDArray outputs = generator.generate(inputIdArray);
            return decodeWithOffset(tokenizer, outputs, generator.getPositionOffset());
        }
    }

    public static String[] generateTextWithPyTorchBeam()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        SearchConfig config = new SearchConfig();
        config.setMaxSeqLength(60);
        long padTokenId = 220;
        config.setPadTokenId(padTokenId);

        String url = "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_pt.zip";

        Criteria<NDList, CausalLMOutput> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, CausalLMOutput.class)
                        .optModelUrls(url)
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new DeferredTranslatorFactory())
                        .build();
        String[] inputs = {"DeepMind Company is", "Memories follow me left and right. I can"};

        try (ZooModel<NDList, CausalLMOutput> model = criteria.loadModel();
                Predictor<NDList, CausalLMOutput> predictor = model.newPredictor();
                NDManager manager = model.getNDManager().newSubManager();
                HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("gpt2")) {

            TextGenerator generator = new TextGenerator(predictor, "beam", config);
            NDArray inputIdArray = encodeWithPadding(manager, tokenizer, inputs, padTokenId);

            NDArray outputs = generator.generate(inputIdArray);
            return decodeWithOffset(
                    tokenizer, outputs, generator.getPositionOffset().repeat(0, config.getBeam()));
        }
    }

    public static String[] generateTextWithOnnxRuntimeBeam()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        SearchConfig config = new SearchConfig();
        config.setMaxSeqLength(60);
        long padTokenId = 220;
        config.setPadTokenId(padTokenId);

        // The model is converted optimum:
        // https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-using-past-keysvalues-in-the-decoder
        /*
         * optimum-cli export onnx --model gpt2 gpt2_onnx/
         *
         * from transformers import AutoTokenizer
         * from optimum.onnxruntime import ORTModelForCausalLM
         *
         * tokenizer = AutoTokenizer.from_pretrained("./gpt2_onnx/")
         * model = ORTModelForCausalLM.from_pretrained("./gpt2_onnx/")
         * inputs = tokenizer("My name is Arthur and I live in", return_tensors="pt")
         * gen_tokens = model.generate(**inputs)
         * print(tokenizer.batch_decode(gen_tokens))
         */
        String url = "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_onnx.zip";

        Criteria<NDList, CausalLMOutput> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, CausalLMOutput.class)
                        .optModelUrls(url)
                        .optEngine("OnnxRuntime")
                        .optTranslatorFactory(new DeferredTranslatorFactory())
                        .build();
        String[] inputs = {"DeepMind Company is"};

        try (ZooModel<NDList, CausalLMOutput> model = criteria.loadModel();
                Predictor<NDList, CausalLMOutput> predictor = model.newPredictor();
                NDManager manager = model.getNDManager().newSubManager();
                HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("gpt2")) {

            TextGenerator generator = new TextGenerator(predictor, "beam", config);
            NDArray inputIdArray = encodeWithPadding(manager, tokenizer, inputs, padTokenId);

            NDArray outputs = generator.generate(inputIdArray);
            return decodeWithOffset(
                    tokenizer, outputs, generator.getPositionOffset().repeat(0, config.getBeam()));
        }
    }

    public static NDArray encodeWithPadding(
            NDManager manager, HuggingFaceTokenizer tokenizer, String[] inputs, long padTokenId) {
        NDArray inputIdArray = null;
        for (String input : inputs) {
            long[] inputIds = tokenizer.encode(input).getIds();
            NDArray deltaInputIdArray = manager.create(inputIds).expandDims(0);
            if (inputIdArray == null) {
                inputIdArray = deltaInputIdArray;
            } else {
                if (inputIdArray.getShape().get(1) > deltaInputIdArray.getShape().get(1)) {
                    // pad deltaInputIdArray
                    long batchSize = deltaInputIdArray.getShape().get(0);
                    long deltaSeqLength =
                            inputIdArray.getShape().get(1) - deltaInputIdArray.getShape().get(1);
                    deltaInputIdArray =
                            manager.full(
                                            new Shape(batchSize, deltaSeqLength),
                                            padTokenId,
                                            DataType.INT64)
                                    .concat(deltaInputIdArray, 1);
                } else if (inputIdArray.getShape().get(1) < deltaInputIdArray.getShape().get(1)) {
                    // pad inputIdArray
                    long batchSize = inputIdArray.getShape().get(0);
                    long deltaSeqLength =
                            deltaInputIdArray.getShape().get(1) - inputIdArray.getShape().get(1);
                    inputIdArray =
                            manager.full(
                                            new Shape(batchSize, deltaSeqLength),
                                            padTokenId,
                                            DataType.INT64)
                                    .concat(inputIdArray, 1);
                }
                inputIdArray = inputIdArray.concat(deltaInputIdArray, 0);
            }
        }
        return inputIdArray;
    }

    public static String[] decodeWithOffset(
            HuggingFaceTokenizer tokenizer, NDArray outputIds, NDArray offset) {
        long batchSize = outputIds.getShape().get(0);
        String[] outputs = new String[(int) batchSize];
        for (int i = 0; i < batchSize; i++) {
            long startIndex = offset.getLong(i);
            long[] outputId = outputIds.get("{},{}:", i, startIndex).toLongArray();
            outputs[i] = tokenizer.decode(outputId);
        }
        return outputs;
    }
}
