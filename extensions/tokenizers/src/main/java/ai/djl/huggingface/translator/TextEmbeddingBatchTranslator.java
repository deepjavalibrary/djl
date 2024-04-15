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
import ai.djl.translate.TranslatorContext;

import java.util.Arrays;

/** The translator for Huggingface text embedding model. */
public class TextEmbeddingBatchTranslator implements NoBatchifyTranslator<String[], float[][]> {

    private HuggingFaceTokenizer tokenizer;
    private Batchifier batchifier;
    private boolean normalize;
    private String pooling;
    private boolean includeTokenTypes;

    TextEmbeddingBatchTranslator(
            HuggingFaceTokenizer tokenizer,
            Batchifier batchifier,
            String pooling,
            boolean normalize,
            boolean includeTokenTypes) {
        this.tokenizer = tokenizer;
        this.batchifier = batchifier;
        this.pooling = pooling;
        this.normalize = normalize;
        this.includeTokenTypes = includeTokenTypes;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String[] input) {
        NDManager manager = ctx.getNDManager();
        Object[] encodings = Arrays.stream(tokenizer.batchEncode(input)).toArray();
        ctx.setAttachment("encodings", encodings);
        NDList[] batch = new NDList[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            batch[i] = ((Encoding) encodings[i]).toNDList(manager, includeTokenTypes);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public float[][] processOutput(TranslatorContext ctx, NDList list) {
        NDList[] batch = batchifier.unbatchify(list);
        Object[] encoding = (Object[]) ctx.getAttachment("encodings");
        NDManager manager = ctx.getNDManager();
        float[][] ret = new float[batch.length][];
        for (int i = 0; i < batch.length; ++i) {
            NDArray array =
                    TextEmbeddingTranslator.processEmbedding(
                            manager, batch[i], (Encoding) encoding[i], pooling);
            if (normalize) {
                array = array.normalize(2, 0);
            }
            ret[i] = array.toFloatArray();
        }

        return ret;
    }
}
