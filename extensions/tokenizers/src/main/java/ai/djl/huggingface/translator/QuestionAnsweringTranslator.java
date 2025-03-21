/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** The translator for Huggingface question answering model. */
public class QuestionAnsweringTranslator implements Translator<QAInput, String> {

    private HuggingFaceTokenizer tokenizer;
    private boolean includeTokenTypes;
    private boolean int32;
    private Batchifier batchifier;
    private boolean detail;

    QuestionAnsweringTranslator(
            HuggingFaceTokenizer tokenizer,
            boolean includeTokenTypes,
            boolean int32,
            Batchifier batchifier,
            boolean detail) {
        this.tokenizer = tokenizer;
        this.includeTokenTypes = includeTokenTypes;
        this.int32 = int32;
        this.batchifier = batchifier;
        this.detail = detail;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) {
        Encoding encoding = tokenizer.encode(input.getQuestion(), input.getParagraph());
        ctx.setAttachment("encoding", encoding);
        return encoding.toNDList(ctx.getNDManager(), includeTokenTypes, int32);
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchProcessInput(TranslatorContext ctx, List<QAInput> inputs) {
        NDManager manager = ctx.getNDManager();
        PairList<String, String> pair = new PairList<>(inputs.size());
        for (QAInput input : inputs) {
            pair.add(input.getQuestion(), input.getParagraph());
        }
        Encoding[] encodings = tokenizer.batchEncode(pair);
        ctx.setAttachment("encodings", encodings);
        NDList[] batch = new NDList[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            batch[i] = encodings[i].toNDList(manager, includeTokenTypes, int32);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        Encoding encoding = (Encoding) ctx.getAttachment("encoding");
        return decode(list, encoding);
    }

    /** {@inheritDoc} */
    @Override
    public List<String> batchProcessOutput(TranslatorContext ctx, NDList list) {
        NDList[] batch = batchifier.unbatchify(list);
        Encoding[] encodings = (Encoding[]) ctx.getAttachment("encodings");
        List<String> ret = new ArrayList<>(batch.length);
        for (int i = 0; i < encodings.length; ++i) {
            ret.add(decode(batch[i], encodings[i]));
        }
        return ret;
    }

    private String decode(NDList list, Encoding encoding) {
        NDArray startLogits = list.get(0);
        NDArray endLogits = list.get(1);
        if ("PyTorch".equals(startLogits.getManager().getEngine().getEngineName())) {
            // PyTorch InferenceMode tensor is read only, must clone it
            startLogits = startLogits.duplicate();
            endLogits = endLogits.duplicate();
        }
        if (detail) {
            // exclude undesired sequences
            long[] sequenceIds = encoding.getSequenceIds();
            List<Integer> undesired = new ArrayList<>();
            for (int i = 0; i < sequenceIds.length; ++i) {
                if (sequenceIds[i] == 0) {
                    undesired.add(i);
                }
            }
            int[] idx = undesired.stream().mapToInt(Integer::intValue).toArray();
            NDIndex ndIndex = new NDIndex("{}", list.getManager().create(idx));
            startLogits.set(ndIndex, -100000f);
            endLogits.set(ndIndex, -100000f);

            // normalize
            startLogits = startLogits.sub(startLogits.max()).exp();
            startLogits = startLogits.div(startLogits.sum());
            endLogits = endLogits.sub(endLogits.max()).exp();
            endLogits = endLogits.div(endLogits.sum());
        }

        // exclude <CLS>, TODO: exclude impossible ids properly and handle max answer length
        startLogits.set(new NDIndex(0), -100000);
        endLogits.set(new NDIndex(0), -100000);
        int startIdx = (int) startLogits.argMax().getLong();
        int endIdx = (int) endLogits.argMax().getLong();
        if (startIdx > endIdx) {
            int tmp = startIdx;
            startIdx = endIdx;
            endIdx = tmp;
            NDArray tmpArray = startLogits;
            startLogits = endLogits;
            endLogits = tmpArray;
        }
        long[] indices = encoding.getIds();
        int len = endIdx - startIdx + 1;
        long[] ids = new long[len];
        System.arraycopy(indices, startIdx, ids, 0, len);
        String answer = tokenizer.decode(ids).trim();
        if (detail) {
            float score = startLogits.getFloat(startIdx) * endLogits.getFloat(endIdx);

            Map<String, Object> dict = new ConcurrentHashMap<>();
            dict.put("score", score);
            dict.put("start", startIdx);
            dict.put("end", endIdx);
            dict.put("answer", answer);
            return JsonUtils.toJson(dict);
        }
        return answer;
    }

    /**
     * Creates a builder to build a {@code QuestionAnsweringTranslator}.
     *
     * @param tokenizer the tokenizer
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer) {
        return new Builder(tokenizer);
    }

    /**
     * Creates a builder to build a {@code QuestionAnsweringTranslator}.
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

    /** The builder for question answering translator. */
    public static final class Builder {

        private HuggingFaceTokenizer tokenizer;
        private boolean includeTokenTypes;
        private boolean int32;
        private Batchifier batchifier = Batchifier.STACK;
        private boolean detail;

        Builder(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        /**
         * Sets if include token types for the {@link Translator}.
         *
         * @param includeTokenTypes true to include token types
         * @return this builder
         */
        public Builder optIncludeTokenTypes(boolean includeTokenTypes) {
            this.includeTokenTypes = includeTokenTypes;
            return this;
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
         * Sets the {@link Batchifier} for the {@link Translator}.
         *
         * @param batchifier true to include token types
         * @return this builder
         */
        public Builder optBatchifier(Batchifier batchifier) {
            this.batchifier = batchifier;
            return this;
        }

        /**
         * Sets if output detail for the {@link Translator}.
         *
         * @param detail true to output detail
         * @return this builder
         */
        public Builder optDetail(boolean detail) {
            this.detail = detail;
            return this;
        }

        /**
         * Configures the builder with the model arguments.
         *
         * @param arguments the model arguments
         */
        public void configure(Map<String, ?> arguments) {
            optIncludeTokenTypes(ArgumentsUtil.booleanValue(arguments, "includeTokenTypes"));
            optInt32(ArgumentsUtil.booleanValue(arguments, "int32"));
            String batchifierStr = ArgumentsUtil.stringValue(arguments, "batchifier", "stack");
            optDetail(ArgumentsUtil.booleanValue(arguments, "detail"));
            optBatchifier(Batchifier.fromString(batchifierStr));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public QuestionAnsweringTranslator build() {
            return new QuestionAnsweringTranslator(
                    tokenizer, includeTokenTypes, int32, batchifier, detail);
        }
    }
}
