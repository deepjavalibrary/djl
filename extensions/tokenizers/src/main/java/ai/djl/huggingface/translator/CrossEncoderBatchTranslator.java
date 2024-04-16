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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.PairList;
import ai.djl.util.StringPair;

import java.util.Arrays;

/** The translator for Huggingface cross encoder model. */
public class CrossEncoderBatchTranslator implements NoBatchifyTranslator<StringPair[], float[][]> {

    private HuggingFaceTokenizer tokenizer;
    private boolean includeTokenTypes;
    private boolean sigmoid;
    private Batchifier batchifier;

    CrossEncoderBatchTranslator(
            HuggingFaceTokenizer tokenizer, boolean includeTokenTypes, boolean sigmoid, Batchifier batchifier) {
        this.tokenizer = tokenizer;
        this.includeTokenTypes = includeTokenTypes;
        this.sigmoid = sigmoid;
        this.batchifier = batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, StringPair[] inputs)
            throws TranslateException {
        NDManager manager = ctx.getNDManager();
        PairList<String, String> list = new PairList<>(Arrays.asList(inputs));
        Encoding[] encodings = tokenizer.batchEncode(list);
        NDList[] batch = new NDList[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            batch[i] = encodings[i].toNDList(manager, includeTokenTypes);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public float[][] processOutput(TranslatorContext ctx, NDList list) {
        NDList[] batch = batchifier.unbatchify(list);
        float[][] ret = new float[batch.length][];
        for (int i = 0; i < batch.length; ++i) {
            NDArray result = list.get(0);
            if (sigmoid) {
                result = result.getNDArrayInternal().sigmoid();
            }
            ret[i] = result.toFloatArray();
        }
        return ret;
    }
}
