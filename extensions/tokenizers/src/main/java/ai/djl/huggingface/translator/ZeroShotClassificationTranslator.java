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
package ai.djl.huggingface.translator;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.translator.ZeroShotClassificationInput;
import ai.djl.modality.nlp.translator.ZeroShotClassificationOutput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;

import com.google.gson.reflect.TypeToken;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** The translator for Huggingface zero-shot-classification model. */
public class ZeroShotClassificationTranslator
        implements NoBatchifyTranslator<ZeroShotClassificationInput, ZeroShotClassificationOutput> {

    private static final Logger logger =
            LoggerFactory.getLogger(ZeroShotClassificationTranslator.class);

    private HuggingFaceTokenizer tokenizer;
    private int entailmentId;
    private int contradictionId;
    private boolean tokenTypeId;
    private boolean int32;
    private Predictor<NDList, NDList> predictor;

    ZeroShotClassificationTranslator(
            HuggingFaceTokenizer tokenizer, boolean tokenTypeId, boolean int32) {
        this.tokenizer = tokenizer;
        this.tokenTypeId = tokenTypeId;
        this.int32 = int32;
    }

    ZeroShotClassificationTranslator(
            HuggingFaceTokenizer tokenizer,
            boolean tokenTypeId,
            boolean int32,
            int entailmentId,
            int contradictionId) {
        this(tokenizer, tokenTypeId, int32);
        this.entailmentId = entailmentId;
        this.contradictionId = contradictionId;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException, ModelException {
        Model model = ctx.getModel();
        predictor = model.newPredictor(new NoopTranslator(null));
        ctx.getPredictorManager().attachInternal(UUID.randomUUID().toString(), predictor);

        URL configUrl = null;
        try {
            configUrl = model.getArtifact("config.json");
        } catch (IOException e) {
            logger.error("Error reading config.json of model: {}", model.getName(), e);
        }

        if (configUrl != null) {
            try (InputStream is = configUrl.openStream();
                    InputStreamReader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {

                Map<String, Object> config =
                        JsonUtils.GSON.fromJson(
                                reader, new TypeToken<Map<String, Object>>() {}.getType());

                if (config != null) {
                    if (config.containsKey("label2id")) {
                        String label2IdJson = JsonUtils.GSON.toJson(config.get("label2id"));
                        Map<String, Double> label2idDouble =
                                JsonUtils.GSON.fromJson(
                                        label2IdJson,
                                        new TypeToken<Map<String, Double>>() {}.getType());

                        Map<String, Integer> label2id = new HashMap<>();
                        label2idDouble.forEach(
                                (k, v) -> label2id.put(k, v != null ? v.intValue() : null));

                        for (Map.Entry<String, Integer> entry : label2id.entrySet()) {
                            if (entry.getKey().toLowerCase().startsWith("entail")
                                    && entry.getValue() != null) {
                                entailmentId = entry.getValue();
                            } else if (entry.getKey().toLowerCase().startsWith("contra")
                                    && entry.getValue() != null) {
                                contradictionId = entry.getValue();
                            }
                        }
                    }

                    boolean inferredWithTokenType = false; // Default assumption

                    if (config.containsKey("type_vocab_size")) {
                        Object typeVocabSizeObj = config.get("type_vocab_size");
                        if (typeVocabSizeObj instanceof Number) {
                            int typeVocabSize = ((Number) typeVocabSizeObj).intValue();
                            if (typeVocabSize > 1) {
                                inferredWithTokenType = true;
                            }
                        }
                    }

                    if (!inferredWithTokenType && config.containsKey("model_type")) {
                        String modelType = (String) config.get("model_type");
                        if (modelType != null) {
                            if ("bert".equals(modelType)
                                    || "albert".equals(modelType)
                                    || "xlnet".equals(modelType)
                                    || "deberta".startsWith(modelType)) {
                                inferredWithTokenType = true;
                            }
                        }
                    }

                    tokenTypeId = inferredWithTokenType;
                }
            } catch (IOException e) {
                logger.error(
                        "Failed to read or parse config.json for label2id: {}", e.getMessage(), e);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, ZeroShotClassificationInput input) {
        ctx.setAttachment("input", input);
        return new NDList();
    }

    /** {@inheritDoc} */
    @Override
    public ZeroShotClassificationOutput processOutput(TranslatorContext ctx, NDList list)
            throws TranslateException {
        ZeroShotClassificationInput input =
                (ZeroShotClassificationInput) ctx.getAttachment("input");

        String template = input.getHypothesisTemplate();
        String[] candidates = input.getCandidates();
        if (candidates == null || candidates.length == 0) {
            throw new TranslateException("Missing candidates in input");
        }

        NDManager manager = ctx.getNDManager();
        NDList output = new NDList(candidates.length);
        for (String candidate : candidates) {
            String hypothesis = applyTemplate(template, candidate);
            Encoding encoding = tokenizer.encode(input.getText(), hypothesis);
            NDList in = encoding.toNDList(manager, tokenTypeId, int32);
            NDList batch = Batchifier.STACK.batchify(new NDList[] {in});
            output.add(predictor.predict(batch).get(0));
        }

        NDArray combinedLogits = NDArrays.concat(output);

        String[] finalLabels;
        double[] finalScores;

        if (input.isMultiLabel() || candidates.length == 1) {
            NDArray entailmentScores;
            if (combinedLogits.getShape().get(1) == 2) {
                // Binary classification: [not entail, entail]
                NDArray probs = combinedLogits.softmax(1);
                entailmentScores = probs.get(":, " + entailmentId);
            } else {
                // 3-class NLI output (e.g., entailment, neutral, contradiction)
                NDArray entailContrLogits =
                        combinedLogits.get(
                                new NDIndex(
                                        ":, {}",
                                        manager.create(new int[] {contradictionId, entailmentId})));
                NDArray scoresProbs = entailContrLogits.softmax(1);
                entailmentScores = scoresProbs.get(":, 1");
            }

            float[] floatScores = entailmentScores.toFloatArray();
            double[] tempScores = new double[floatScores.length];
            for (int i = 0; i < floatScores.length; i++) {
                tempScores[i] = floatScores[i];
            }

            List<AbstractMap.SimpleEntry<Double, String>> pairs = new ArrayList<>();
            for (int i = 0; i < candidates.length; i++) {
                pairs.add(new AbstractMap.SimpleEntry<>(tempScores[i], candidates[i]));
            }

            pairs.sort(
                    Comparator.comparingDouble(
                                    (AbstractMap.SimpleEntry<Double, String> e) -> e.getKey())
                            .reversed());

            finalLabels = new String[candidates.length];
            finalScores = new double[candidates.length];
            for (int i = 0; i < candidates.length; i++) {
                finalLabels[i] = pairs.get(i).getValue();
                finalScores[i] = pairs.get(i).getKey();
            }

        } else { // Single-label classification (len(candidate_labels) > 1 and not multi_label)
            NDArray entailLogits = combinedLogits.get(":, " + entailmentId);
            NDArray exp = entailLogits.exp();
            NDArray sum = exp.sum();
            NDArray normalizedScores = exp.div(sum); // Probabilities sum to 1 across candidates

            long[] indices = normalizedScores.argSort(-1, false).toLongArray();
            float[] probabilities = normalizedScores.toFloatArray();

            finalLabels = new String[candidates.length];
            finalScores = new double[candidates.length];
            for (int i = 0; i < finalLabels.length; ++i) {
                int index = (int) indices[i];
                finalLabels[i] = candidates[index];
                finalScores[i] = probabilities[index];
            }
        }

        return new ZeroShotClassificationOutput(input.getText(), finalLabels, finalScores);
    }

    private String applyTemplate(String template, String arg) {
        int pos = template.indexOf("{}");
        if (pos == -1) {
            return template + arg;
        }
        int len = template.length();
        return template.substring(0, pos) + arg + template.substring(pos + 2, len);
    }

    /**
     * Creates a builder to build a {@code ZeroShotClassificationTranslator}.
     *
     * @param tokenizer the tokenizer
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer) {
        return new Builder(tokenizer);
    }

    /**
     * Creates a builder to build a {@code ZeroShotClassificationTranslator}.
     *
     * @param tokenizer the tokenizer
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static ZeroShotClassificationTranslator.Builder builder(
            HuggingFaceTokenizer tokenizer, Map<String, ?> arguments) {
        ZeroShotClassificationTranslator.Builder builder = builder(tokenizer);
        builder.configure(arguments);

        return builder;
    }

    /** The builder for zero-shot classification translator. */
    public static final class Builder {

        private HuggingFaceTokenizer tokenizer;
        private boolean tokenTypeId;
        private boolean int32;
        private int entailmentId = 2;
        private int contradictionId;

        Builder(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        /**
         * Specifies whether to include token type IDs in the input tensors.
         *
         * @param tokenTypeId {@code true} to include token type IDs, {@code false} to omit them
         * @return this builder instance for method chaining
         */
        public Builder optTokenTypeId(boolean tokenTypeId) {
            this.tokenTypeId = tokenTypeId;
            return this;
        }

        /**
         * Specifies whether to use int32 as the data type for input token tensors.
         *
         * @param int32 {@code true} to use int32 inputs, {@code false} to use the default type
         * @return this builder instance for method chaining
         */
        public Builder optInt32(boolean int32) {
            this.int32 = int32;
            return this;
        }

        /**
         * Optional: Set custom entailment ID if different from default (2). This value usually
         * comes from the model's `config.json` `label2id` mapping.
         *
         * @param entailmentId The index for the 'entailment' label.
         * @return this builder
         */
        public Builder optEntailmentId(int entailmentId) {
            this.entailmentId = entailmentId;
            return this;
        }

        /**
         * Optional: Set custom contradiction ID if different from default (0). This value usually
         * comes from the model's `config.json` `label2id` mapping.
         *
         * @param contradictionId The index for the 'contradiction' label.
         * @return this builder
         */
        public Builder optContradictionId(int contradictionId) {
            this.contradictionId = contradictionId;
            return this;
        }

        /**
         * Configures the builder with the model arguments.
         *
         * @param arguments the model arguments
         */
        public void configure(Map<String, ?> arguments) {
            optTokenTypeId(ArgumentsUtil.booleanValue(arguments, "tokenTypeId"));
            optInt32(ArgumentsUtil.booleanValue(arguments, "int32"));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public ZeroShotClassificationTranslator build() throws IOException {
            return new ZeroShotClassificationTranslator(
                    tokenizer, tokenTypeId, int32, entailmentId, contradictionId);
        }
    }
}
