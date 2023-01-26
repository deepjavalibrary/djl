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
package ai.djl.huggingface.translator;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.PairList;

/** The translator for Huggingface question answering model. */
public class QuestionAnsweringBatchTranslator implements NoBatchifyTranslator<QAInput[], String[]> {

    private HuggingFaceTokenizer tokenizer;
    private boolean includeTokenTypes;
    private Batchifier batchifier;

    QuestionAnsweringBatchTranslator(
            HuggingFaceTokenizer tokenizer, boolean includeTokenTypes, Batchifier batchifier) {
        this.tokenizer = tokenizer;
        this.includeTokenTypes = includeTokenTypes;
        this.batchifier = batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, QAInput[] inputs) {
        NDManager manager = ctx.getNDManager();
        PairList<String, String> pair = new PairList<>(inputs.length);
        for (QAInput input : inputs) {
            pair.add(input.getQuestion(), input.getParagraph());
        }
        Encoding[] encodings = tokenizer.batchEncode(pair);
        ctx.setAttachment("encodings", encodings);
        NDList[] batch = new NDList[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            batch[i] = encodings[i].toNDList(manager, includeTokenTypes);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public String[] processOutput(TranslatorContext ctx, NDList list) {
        NDList[] batch = batchifier.unbatchify(list);
        Encoding[] encodings = (Encoding[]) ctx.getAttachment("encodings");
        String[] ret = new String[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            ret[i] = QuestionAnsweringTranslator.decode(batch[i], encodings[i], tokenizer);
        }
        return ret;
    }
}
