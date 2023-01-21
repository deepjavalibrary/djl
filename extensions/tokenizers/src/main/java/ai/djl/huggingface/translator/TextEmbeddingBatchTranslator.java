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

/** The translator for Huggingface text embedding model. */
public class TextEmbeddingBatchTranslator implements NoBatchifyTranslator<String[], float[][]> {

    private HuggingFaceTokenizer tokenizer;
    private Batchifier batchifier;
    private boolean normalize;
    private String pooling;

    TextEmbeddingBatchTranslator(
            HuggingFaceTokenizer tokenizer,
            Batchifier batchifier,
            String pooling,
            boolean normalize) {
        this.tokenizer = tokenizer;
        this.batchifier = batchifier;
        this.pooling = pooling;
        this.normalize = normalize;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String[] input) {
        NDManager manager = ctx.getNDManager();
        Encoding[] encodings = tokenizer.batchEncode(input);
        ctx.setAttachment("encoding", encodings);
        NDList[] batch = new NDList[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            batch[i] = encodings[i].toNDList(manager, false);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public float[][] processOutput(TranslatorContext ctx, NDList list) {
        NDArray embeddings = list.get("last_hidden_state");
        NDList[] batch = batchifier.unbatchify(new NDList(embeddings));
        Encoding[] encoding = (Encoding[]) ctx.getAttachment("encoding");
        NDManager manager = ctx.getNDManager();
        float[][] ret = new float[batch.length][];
        for (int i = 0; i < batch.length; ++i) {
            NDArray embedding = batch[i].singletonOrThrow();
            NDArray array =
                    TextEmbeddingTranslator.processEmbedding(
                            manager, embedding, encoding[i], pooling);
            if (normalize) {
                array = array.normalize(2, 0);
            }
            ret[i] = array.toFloatArray();
        }

        return ret;
    }
}
