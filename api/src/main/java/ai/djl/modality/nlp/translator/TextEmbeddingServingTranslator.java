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
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.List;

/** A {@link Translator} that can handle generic text embedding {@link Input} and {@link Output}. */
public class TextEmbeddingServingTranslator implements Translator<Input, Output> {

    private Translator<String, float[]> translator;

    /**
     * Constructs a {@code TextEmbeddingServingTranslator} instance.
     *
     * @param translator a {@code Translator} processes text embedding input
     */
    public TextEmbeddingServingTranslator(Translator<String, float[]> translator) {
        this.translator = translator;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        translator.prepare(ctx);
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
            return translator.batchProcessInput(ctx, prompt.getBatch());
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
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    public NDList batchProcessInput(TranslatorContext ctx, List<Input> inputs) throws Exception {
        int[] mapping = new int[inputs.size()];
        List<String> prompts = new ArrayList<>(mapping.length);
        for (int i = 0; i < mapping.length; ++i) {
            TextPrompt prompt = TextPrompt.parseInput(inputs.get(i));
            if (prompt.isBatch()) {
                List<String> batch = prompt.getBatch();
                mapping[i] = batch.size();
                prompts.addAll(batch);
            } else {
                mapping[i] = -1;
                prompts.add(prompt.getText());
            }
        }
        ctx.setAttachment("mapping", mapping);
        return translator.batchProcessInput(ctx, prompts);
    }

    /** {@inheritDoc} */
    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) throws Exception {
        Output output = new Output();
        output.addProperty("Content-Type", "application/json");
        if (ctx.getAttachment("batch") != null) {
            output.add(BytesSupplier.wrapAsJson(translator.batchProcessOutput(ctx, list)));
        } else {
            Batchifier batchifier = translator.getBatchifier();
            if (batchifier != null) {
                list = batchifier.unbatchify(list)[0];
            }
            output.add(BytesSupplier.wrapAsJson(translator.processOutput(ctx, list)));
        }
        return output;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    public List<Output> batchProcessOutput(TranslatorContext ctx, NDList list) throws Exception {
        List<float[]> outputs = translator.batchProcessOutput(ctx, list);
        int[] mapping = (int[]) ctx.getAttachment("mapping");
        List<Output> ret = new ArrayList<>(mapping.length);
        int index = 0;
        for (int size : mapping) {
            Output output = new Output();
            output.addProperty("Content-Type", "application/json");
            if (size == -1) {
                // non-batching
                output.add(BytesSupplier.wrapAsJson(outputs.get(index++)));
            } else {
                // client side batching
                float[][] embeddings = new float[size][];
                for (int j = 0; j < size; ++j) {
                    embeddings[j] = outputs.get(index++);
                }
                output.add(BytesSupplier.wrapAsJson(embeddings));
            }
            ret.add(output);
        }
        return ret;
    }
}
