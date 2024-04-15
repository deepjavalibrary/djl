/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.nlp.translator;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.nlp.TextPrompt;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/** A {@link Translator} that can handle generic text embedding {@link Input} and {@link Output}. */
public class TextEmbeddingServingTranslator implements NoBatchifyTranslator<Input, Output> {

    private Translator<String, float[]> translator;
    private Translator<String[], float[][]> batchTranslator;

    /**
     * Constructs a {@code TextEmbeddingServingTranslator} instance.
     *
     * @param translator a {@code Translator} processes text embedding input
     */
    public TextEmbeddingServingTranslator(Translator<String, float[]> translator) {
        this.translator = translator;
        this.batchTranslator = translator.toBatchTranslator();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        translator.prepare(ctx);
        batchTranslator.prepare(ctx);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
        if (input.getContent().isEmpty()) {
            throw new TranslateException("Input data is empty.");
        }

        TextPrompt prompt = TextPrompt.parseInput(input);
        if (prompt.isBatch()) {
            ctx.setAttachment("batch", Boolean.TRUE);
            return batchTranslator.processInput(ctx, prompt.getBatch());
        }

        NDList ret = translator.processInput(ctx, prompt.getText());
        Batchifier batchifier = translator.getBatchifier();
        if (batchifier != null) {
            NDList[] batch = {ret};
            return batchifier.batchify(batch);
        }
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) throws Exception {
        Output output = new Output();
        output.addProperty("Content-Type", "application/json");
        if (ctx.getAttachment("batch") != null) {
            output.add(BytesSupplier.wrapAsJson(batchTranslator.processOutput(ctx, list)));
        } else {
            Batchifier batchifier = translator.getBatchifier();
            if (batchifier != null) {
                list = batchifier.unbatchify(list)[0];
            }
            output.add(BytesSupplier.wrapAsJson(translator.processOutput(ctx, list)));
        }
        return output;
    }
}
