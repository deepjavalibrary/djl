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
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.util.Map;
import java.util.UUID;

/** The translator for Huggingface zero-shot-classification model. */
public class ZeroShotClassificationTranslator
        implements NoBatchifyTranslator<ZeroShotClassificationInput, ZeroShotClassificationOutput> {

    private HuggingFaceTokenizer tokenizer;
    private boolean int32;
    private Predictor<NDList, NDList> predictor;

    ZeroShotClassificationTranslator(HuggingFaceTokenizer tokenizer, boolean int32) {
        this.tokenizer = tokenizer;
        this.int32 = int32;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException, ModelException {
        Model model = ctx.getModel();
        predictor = model.newPredictor(new NoopTranslator(null));
        ctx.getPredictorManager().attachInternal(UUID.randomUUID().toString(), predictor);
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
            NDList in = encoding.toNDList(manager, false, int32);
            NDList batch = Batchifier.STACK.batchify(new NDList[] {in});
            output.add(predictor.predict(batch).get(0));
        }

        NDArray logits = NDArrays.concat(output);
        if (input.isMultiLabel()) {
            logits = logits.get(":, -1");
            logits = logits.softmax(-1);
        } else {
            logits = logits.get(new NDIndex(":, {}", manager.create(new int[] {0, 2})));
            logits = logits.softmax(1);
            logits = logits.get(":, -1");
        }
        long[] indices = logits.argSort(-1, false).toLongArray();
        float[] probabilities = logits.toFloatArray();

        String[] labels = new String[candidates.length];
        double[] scores = new double[candidates.length];
        for (int i = 0; i < labels.length; ++i) {
            int index = (int) indices[i];
            labels[i] = candidates[index];
            scores[i] = probabilities[index];
        }

        return new ZeroShotClassificationOutput(input.getText(), labels, scores);
    }

    private String applyTemplate(String template, String arg) {
        int pos = template.indexOf("{}");
        if (pos == -1) {
            return template + arg;
        }
        int len = template.length();
        return template.substring(0, pos) + arg + template.substring(pos + 1, len);
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
    public static Builder builder(HuggingFaceTokenizer tokenizer, Map<String, ?> arguments) {
        Builder builder = builder(tokenizer);
        builder.configure(arguments);

        return builder;
    }

    /** The builder for zero-shot classification translator. */
    public static final class Builder {

        private HuggingFaceTokenizer tokenizer;
        private boolean int32;

        Builder(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        /**
         * Sets if use int32 datatype for the {@link Translator}.
         *
         * @param int32 true to include token types
         * @return this builder
         */
        public Builder optInt32(boolean int32) {
            this.int32 = int32;
            return this;
        }

        /**
         * Configures the builder with the model arguments.
         *
         * @param arguments the model arguments
         */
        public void configure(Map<String, ?> arguments) {
            optInt32(ArgumentsUtil.booleanValue(arguments, "int32"));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public ZeroShotClassificationTranslator build() throws IOException {
            return new ZeroShotClassificationTranslator(tokenizer, int32);
        }
    }
}
