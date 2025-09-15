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
import ai.djl.huggingface.tokenizers.jni.CharSpan;
import ai.djl.modality.nlp.translator.NamedEntity;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.Pair;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/** The translator for Huggingface token classification model. */
public class TokenClassificationTranslator implements Translator<String, NamedEntity[]> {

    private HuggingFaceTokenizer tokenizer;
    private boolean includeTokenTypes;
    private boolean int32;
    private boolean softmax;
    private String aggregationStrategy;
    private Batchifier batchifier;
    private PretrainedConfig config;

    TokenClassificationTranslator(Builder builder) {
        this.tokenizer = builder.tokenizer;
        this.includeTokenTypes = builder.includeTokenTypes;
        this.int32 = builder.int32;
        this.softmax = builder.softmax;
        this.aggregationStrategy = builder.aggregationStrategy;
        this.batchifier = builder.batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        Path path = ctx.getModel().getModelPath();
        Path file = path.resolve("config.json");
        try (Reader reader = Files.newBufferedReader(file)) {
            config = JsonUtils.GSON.fromJson(reader, PretrainedConfig.class);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        Encoding encoding = tokenizer.encode(input);
        ctx.setAttachment("encoding", encoding);
        ctx.setAttachment("sentence", input);
        return encoding.toNDList(ctx.getNDManager(), includeTokenTypes, int32);
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchProcessInput(TranslatorContext ctx, List<String> inputs) {
        NDManager manager = ctx.getNDManager();
        Encoding[] encodings = tokenizer.batchEncode(inputs);
        ctx.setAttachment("encodings", encodings);
        ctx.setAttachment("sentences", inputs);
        NDList[] batch = new NDList[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            batch[i] = encodings[i].toNDList(manager, includeTokenTypes, int32);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public NamedEntity[] processOutput(TranslatorContext ctx, NDList list) {
        Encoding encoding = (Encoding) ctx.getAttachment("encoding");
        String sentence = (String) ctx.getAttachment("sentence");
        return toNamedEntities(encoding, list, sentence);
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public List<NamedEntity[]> batchProcessOutput(TranslatorContext ctx, NDList list) {
        NDList[] batch = batchifier.unbatchify(list);
        Encoding[] encodings = (Encoding[]) ctx.getAttachment("encodings");
        List<String> sentences = (List<String>) ctx.getAttachment("sentences");
        List<NamedEntity[]> ret = new ArrayList<>(batch.length);
        for (int i = 0; i < batch.length; ++i) {
            ret.add(toNamedEntities(encodings[i], batch[i], sentences.get(i)));
        }
        return ret;
    }

    /**
     * Creates a builder to build a {@code TokenClassificationTranslator}.
     *
     * @param tokenizer the tokenizer
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer) {
        return new Builder(tokenizer);
    }

    /**
     * Creates a builder to build a {@code TokenClassificationTranslator}.
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

    private NamedEntity[] toNamedEntities(Encoding encoding, NDList list, String sentence) {
        long[] inputIds = encoding.getIds();
        CharSpan[] offsetMapping = encoding.getCharTokenSpans();
        long[] specialTokenMasks = encoding.getSpecialTokenMask();
        String[] words = encoding.getTokens();
        long[] tokenIds = encoding.getIds();
        NDArray probabilities = list.get(0);
        if (softmax) {
            probabilities = probabilities.softmax(1);
        }

        List<NamedEntityEx> entities = new ArrayList<>();
        for (int i = 0; i < inputIds.length; ++i) {
            if (specialTokenMasks[i] != 0) {
                continue;
            }

            NDArray prob = probabilities.get(i);
            int start = offsetMapping[i].getStart();
            int end = offsetMapping[i].getEnd();
            boolean isSubWord = false;
            if (start > 0
                    && ("first".equals(aggregationStrategy)
                            || "average".equals(aggregationStrategy)
                            || "max".equals(aggregationStrategy))) {
                int pos = sentence.indexOf(' ', start - 1);
                if (pos < 0 || pos > start) {
                    isSubWord = true;
                }
            }

            NamedEntityEx item =
                    new NamedEntityEx(prob, i, words[i], start, end, tokenIds[i], isSubWord);
            entities.add(item);
        }
        if ("first".equals(aggregationStrategy)
                || "average".equals(aggregationStrategy)
                || "max".equals(aggregationStrategy)) {
            entities = aggregateWords(entities);
            entities = groupEntities(entities);
        } else if ("simple".equals(aggregationStrategy)) {
            entities = groupEntities(entities);
        }

        return entities.stream()
                .filter(o -> !"O".equals(o.getEntity()))
                .map(NamedEntityEx::toNamedEntity)
                .toArray(NamedEntity[]::new);
    }

    private List<NamedEntityEx> aggregateWords(List<NamedEntityEx> entities) {
        List<NamedEntityEx> agg = new ArrayList<>();
        List<NamedEntityEx> group = new ArrayList<>();
        for (NamedEntityEx entity : entities) {
            if (!entity.isSubWord && !group.isEmpty()) {
                agg.add(aggregateWord(group));
                group.clear();
            }
            group.add(entity);
        }
        if (!group.isEmpty()) {
            agg.add(aggregateWord(group));
        }
        return agg;
    }

    private NamedEntityEx aggregateWord(List<NamedEntityEx> entities) {
        if (entities.size() == 1) {
            return entities.get(0);
        }
        List<Long> tokenIds = new ArrayList<>();
        for (NamedEntityEx entity : entities) {
            tokenIds.addAll(entity.tokenIds);
        }
        NamedEntityEx first = entities.get(0);
        NamedEntityEx last = entities.get(entities.size() - 1);

        String entityName;
        float score;

        if ("first".equals(aggregationStrategy)) {
            entityName = first.getEntity();
            score = first.getScore();
        } else if ("max".equals(aggregationStrategy)) {
            NamedEntityEx max =
                    entities.stream()
                            .max(Comparator.comparingDouble(NamedEntityEx::getScore))
                            .get();
            entityName = max.getEntity();
            score = max.getScore();
        } else {
            // average
            NDArray[] arrays = entities.stream().map(o -> o.prob).toArray(NDArray[]::new);
            NDList list = new NDList(arrays);
            NDArray array = NDArrays.stack(list).mean(new int[] {0});
            int entityIdx = (int) array.argMax().getLong();
            entityName = config.id2label.get(String.valueOf(entityIdx));
            score = array.getFloat(entityIdx);
        }
        return new NamedEntityEx(entityName, score, first.start, last.end, tokenIds);
    }

    private List<NamedEntityEx> groupEntities(List<NamedEntityEx> entities) {
        List<NamedEntityEx> disaggregateGroup = new ArrayList<>();
        List<NamedEntityEx> entityGroups = new ArrayList<>();

        for (NamedEntityEx entity : entities) {
            if (disaggregateGroup.isEmpty()) {
                disaggregateGroup.add(entity);
                continue;
            }

            Pair<String, String> tag = getTag(entity.getEntity());
            NamedEntityEx lastEntity = disaggregateGroup.get(disaggregateGroup.size() - 1);
            Pair<String, String> lastTag = getTag(lastEntity.getEntity());
            if (!tag.getValue().equals(lastTag.getValue()) || "B".equals(tag.getKey())) {
                entityGroups.add(groupSubEntities(disaggregateGroup));
                disaggregateGroup.clear();
            }
            disaggregateGroup.add(entity);
        }

        if (!disaggregateGroup.isEmpty()) {
            entityGroups.add(groupSubEntities(disaggregateGroup));
        }
        return entityGroups;
    }

    private Pair<String, String> getTag(String entityName) {
        if (entityName.startsWith("B-")) {
            return new Pair<>("B", entityName.substring(2));
        } else if (entityName.startsWith("I-")) {
            return new Pair<>("I", entityName.substring(2));
        } else {
            return new Pair<>("I", entityName);
        }
    }

    private NamedEntityEx groupSubEntities(List<NamedEntityEx> entities) {
        List<Long> tokens = new ArrayList<>();
        double[] scores = new double[entities.size()];
        for (int i = 0; i < scores.length; ++i) {
            NamedEntityEx entity = entities.get(i);
            tokens.addAll(entity.tokenIds);
            scores[i] = entity.getScore();
        }
        long[] tokenIds = tokens.stream().mapToLong(Long::longValue).toArray();
        String aggWord = tokenizer.decode(tokenIds);
        float aggScore = (float) Arrays.stream(scores).sum() / scores.length;
        NamedEntityEx first = entities.get(0);
        NamedEntityEx last = entities.get(entities.size() - 1);
        String entityName = first.getEntity();
        int pos = entityName.indexOf('-');
        if (pos > 0) {
            entityName = entityName.substring(pos + 1);
        }

        return new NamedEntityEx(entityName, aggScore, aggWord, first.start, last.end);
    }

    /** The builder for token classification translator. */
    public static final class Builder {

        HuggingFaceTokenizer tokenizer;
        boolean includeTokenTypes;
        boolean int32;
        boolean softmax = true;
        String aggregationStrategy;
        Batchifier batchifier = Batchifier.STACK;

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
         * Sets if implement softmax operation for the {@link Translator}.
         *
         * @param softmax true to implement softmax to model output result
         * @return this builder
         */
        public Builder optSoftmax(boolean softmax) {
            this.softmax = softmax;
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
         * Sets the aggregation strategy for the {@link Translator}.
         *
         * @param aggregationStrategy the aggregation strategy, one of none, simple, first, average,
         *     max
         * @return this builder
         */
        public Builder optAggregationStrategy(String aggregationStrategy) {
            this.aggregationStrategy = aggregationStrategy;
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
            optSoftmax(ArgumentsUtil.booleanValue(arguments, "softmax", true));
            optAggregationStrategy(
                    ArgumentsUtil.stringValue(arguments, "aggregation_strategy", "none"));
            String batchifierStr = ArgumentsUtil.stringValue(arguments, "batchifier", "stack");
            optBatchifier(Batchifier.fromString(batchifierStr));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public TokenClassificationTranslator build() {
            return new TokenClassificationTranslator(this);
        }
    }

    private class NamedEntityEx {

        String entity;
        float score;
        int index;
        String word;
        int start;
        int end;
        List<Long> tokenIds;
        boolean isSubWord;
        NDArray prob;
        private boolean initialized;

        NamedEntityEx(String entity, float score, String word, int start, int end) {
            this.entity = entity;
            this.score = score;
            this.index = -1;
            this.word = word;
            this.start = start;
            this.end = end;
            initialized = true;
        }

        NamedEntityEx(String entity, float score, int start, int end, List<Long> tokenIds) {
            this.entity = entity;
            this.score = score;
            this.index = -1;
            this.start = start;
            this.end = end;
            this.tokenIds = tokenIds;
            initialized = true;
        }

        NamedEntityEx(
                NDArray prob,
                int index,
                String word,
                int start,
                int end,
                long tokenId,
                boolean isSubWord) {
            this.prob = prob;
            this.index = index;
            this.word = word;
            this.start = start;
            this.end = end;
            this.tokenIds = Collections.singletonList(tokenId);
            this.isSubWord = isSubWord;
        }

        private void init() {
            if (!initialized) {
                int entityIdx = (int) prob.argMax().getLong();
                entity = config.id2label.get(String.valueOf(entityIdx));
                score = prob.getFloat(entityIdx);
                initialized = true;
            }
        }

        String getEntity() {
            init();
            return entity;
        }

        float getScore() {
            init();
            return score;
        }

        NamedEntity toNamedEntity() {
            init();
            return new NamedEntity(entity, score, index, word, start, end);
        }
    }
}
