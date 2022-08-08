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
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** The translator for Huggingface fill mask model. */
public class FillMaskTranslator implements Translator<String, Classifications> {

    private HuggingFaceTokenizer tokenizer;
    private String maskToken;
    private long maskTokenId;
    private int topK;
    private Batchifier batchifier;

    FillMaskTranslator(
            HuggingFaceTokenizer tokenizer, String maskToken, int topK, Batchifier batchifier) {
        this.tokenizer = tokenizer;
        this.maskToken = maskToken;
        this.topK = topK;
        this.batchifier = batchifier;
        Encoding encoding = tokenizer.encode(maskToken, false);
        maskTokenId = encoding.getIds()[0];
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) throws TranslateException {
        NDManager manager = ctx.getNDManager();
        Encoding encoding = tokenizer.encode(input);
        long[] indices = encoding.getIds();
        int maskIndex = -1;
        for (int i = 0; i < indices.length; ++i) {
            if (indices[i] == maskTokenId) {
                if (maskIndex != -1) {
                    throw new TranslateException("Only one mask supported.");
                }
                maskIndex = i;
            }
        }
        if (maskIndex == -1) {
            throw new TranslateException("Mask token " + maskToken + " not found.");
        }
        ctx.setAttachment("maskIndex", maskIndex);
        long[] attentionMask = encoding.getAttentionMask();
        NDList ndList = new NDList(2);
        ndList.add(manager.create(indices));
        ndList.add(manager.create(attentionMask));
        return ndList;
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        int maskIndex = (int) ctx.getAttachment("maskIndex");
        NDArray prob = list.get(0).get(maskIndex).softmax(0);
        NDArray array = prob.argSort(0, false);
        long[] classIds = new long[topK];
        List<Double> probabilities = new ArrayList<>(topK);
        for (int i = 0; i < topK; ++i) {
            classIds[i] = array.getLong(i);
            probabilities.add((double) prob.getFloat(classIds[i]));
        }
        String[] classes = tokenizer.decode(classIds).trim().split(" ");
        return new Classifications(Arrays.asList(classes), probabilities);
    }

    /**
     * Creates a builder to build a {@code FillMaskTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code FillMaskTranslator}.
     *
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = builder();
        builder.configure(arguments);

        return builder;
    }

    /** The builder for fill mask translator. */
    public static final class Builder {

        private String maskedToken = "[MASK]";
        private int topK = 5;
        private String tokenizerName;
        private Path tokenizerPath;
        private boolean addSpecialTokens = true;
        private Batchifier batchifier = Batchifier.STACK;

        /**
         * Sets the id of the mask {@link Translator}.
         *
         * @param maskedToken the id of the mask
         * @return this builder
         */
        public Builder optMaskToken(String maskedToken) {
            this.maskedToken = maskedToken;
            return this;
        }

        /**
         * Set the topK number of classes to be displayed.
         *
         * @param topK the number of top classes to return
         * @return this builder
         */
        public Builder optTopK(int topK) {
            this.topK = topK;
            return this;
        }

        /**
         * Sets the name of the tokenizer for the {@link Translator}.
         *
         * @param tokenizerName the name of the tokenizer
         * @return this builder
         */
        public Builder optTokenizerName(String tokenizerName) {
            this.tokenizerName = tokenizerName;
            return this;
        }

        /**
         * Sets the file path of the tokenizer for the {@link Translator}.
         *
         * @param tokenizerPath the path of the tokenizer
         * @return this builder
         */
        public Builder optTokenizerPath(Path tokenizerPath) {
            this.tokenizerPath = tokenizerPath;
            return this;
        }

        /**
         * Sets if add special tokens for the {@link Translator}.
         *
         * @param addSpecialTokens true to add special tokens
         * @return this builder
         */
        public Builder optAddSpecialTokens(boolean addSpecialTokens) {
            this.addSpecialTokens = addSpecialTokens;
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
         * Configures the builder with the model arguments.
         *
         * @param arguments the model arguments
         */
        public void configure(Map<String, ?> arguments) {
            optMaskToken(ArgumentsUtil.stringValue(arguments, "maskToken", "[MASK]"));
            optTopK(ArgumentsUtil.intValue(arguments, "topK", 5));
            optTokenizerName(ArgumentsUtil.stringValue(arguments, "tokenizer"));
            optAddSpecialTokens(ArgumentsUtil.booleanValue(arguments, "addSpecialTokens", true));
            String batchifierStr = ArgumentsUtil.stringValue(arguments, "batchifier", "stack");
            optBatchifier(Batchifier.fromString(batchifierStr));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public FillMaskTranslator build() throws IOException {
            HuggingFaceTokenizer tokenizer;
            Map<String, String> options = new ConcurrentHashMap<>();
            options.put("addSpecialTokens", String.valueOf(addSpecialTokens));
            if (tokenizerName != null) {
                tokenizer = HuggingFaceTokenizer.newInstance(tokenizerName, options);
            } else {
                tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath, options);
            }
            return new FillMaskTranslator(tokenizer, maskedToken, topK, batchifier);
        }
    }
}
