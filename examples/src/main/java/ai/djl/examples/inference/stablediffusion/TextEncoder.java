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
package ai.djl.examples.inference.stablediffusion;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;

public class TextEncoder implements NoBatchifyTranslator<String, NDList> {

    private static final int MAX_LENGTH = 77;

    HuggingFaceTokenizer tokenizer;

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        tokenizer =
                HuggingFaceTokenizer.builder()
                        .optPadding(true)
                        .optPadToMaxLength()
                        .optMaxLength(MAX_LENGTH)
                        .optTruncation(true)
                        .optTokenizerName("openai/clip-vit-base-patch32")
                        .build();
    }

    /** {@inheritDoc} */
    @Override
    public NDList processOutput(TranslatorContext ctx, NDList list) {
        list.detach();
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        Encoding encoding = tokenizer.encode(input);
        Shape shape = new Shape(1, encoding.getIds().length);
        return new NDList(ctx.getNDManager().create(encoding.getIds(), shape));
    }
}
