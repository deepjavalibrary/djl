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
import ai.djl.util.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** A {@link Translator} that can handle generic text embedding {@link Input} and {@link Output}. */
public class TextEmbeddingServingTranslator implements Translator<Input, Output> {

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

    /** {@inheritDoc} */
    @Override
    public Translator<Input[], Output[]> toBatchTranslator(Batchifier batchifier) {
        return new NoBatchifyTranslator<Input[], Output[]>() {

            /** {@inheritDoc} */
            @Override
            @SuppressWarnings("PMD.SignatureDeclareThrowsException")
            public NDList processInput(TranslatorContext ctx, Input[] inputs) throws Exception {
                List<String> prompts = new ArrayList<>(inputs.length);
                int[] mapping = new int[inputs.length];
                for (int i = 0; i < inputs.length; ++i) {
                    TextPrompt prompt = TextPrompt.parseInput(inputs[i]);
                    if (prompt.isBatch()) {
                        String[] batch = prompt.getBatch();
                        mapping[i] = batch.length;
                        prompts.addAll(Arrays.asList(batch));
                    } else {
                        mapping[i] = -1;
                        prompts.add(prompt.getText());
                    }
                }
                ctx.setAttachment("mapping", mapping);
                return batchTranslator.processInput(ctx, prompts.toArray(Utils.EMPTY_ARRAY));
            }

            /** {@inheritDoc} */
            @Override
            @SuppressWarnings({"PMD.SignatureDeclareThrowsException", "unchecked"})
            public Output[] processOutput(TranslatorContext ctx, NDList list) throws Exception {
                NDList[] unbatched = batchifier.unbatchify(list);
                int[] mapping = (int[]) ctx.getAttachment("mapping");
                Object[] encodings = (Object[]) ctx.getAttachment("encodings");
                Output[] ret = new Output[mapping.length];
                int index = 0;
                for (int i = 0; i < ret.length; ++i) {
                    Output output = new Output();
                    output.addProperty("Content-Type", "application/json");
                    if (mapping[i] == -1) {
                        // non-batching
                        ctx.setAttachment("encoding", encodings[index]);
                        float[] embedding = translator.processOutput(ctx, unbatched[index]);
                        ++index;
                        output.add(BytesSupplier.wrapAsJson(embedding));
                    } else {
                        float[][] embeddings = new float[mapping[i]][];
                        for (int j = 0; j < mapping[i]; ++j) {
                            ctx.setAttachment("encoding", encodings[index]);
                            embeddings[j] = translator.processOutput(ctx, unbatched[index]);
                            ++index;
                        }
                        output.add(BytesSupplier.wrapAsJson(embeddings));
                    }
                    ret[i] = output;
                }
                return ret;
            }
        };
    }
}
