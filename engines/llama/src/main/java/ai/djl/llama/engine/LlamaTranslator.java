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
package ai.djl.llama.engine;

import ai.djl.inference.streaming.IteratorBytesSupplier;
import ai.djl.llama.jni.InputParameters;
import ai.djl.llama.jni.LlamaLibrary;
import ai.djl.llama.jni.Token;
import ai.djl.llama.jni.TokenIterator;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;

import java.util.Iterator;

/** Built-in {@code Translator} that provides preprocessing and postprocessing for llama.cpp. */
public class LlamaTranslator<I, O> implements NoBatchifyTranslator<I, O> {

    private long handle;

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) {
        LlamaModel model = (LlamaModel) ctx.getModel();
        handle = model.getHandle();
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, I input) {
        if (input instanceof String) {
            ctx.setAttachment("out", generate((String) input));
        } else if (input instanceof LlamaInput) {
            ctx.setAttachment("out", generate((LlamaInput) input));
        } else if (input instanceof Input) {
            String prompt = ((Input) input).getData().getAsString();
            TokenIterator it = generate(prompt);
            Output output = new Output();
            output.add(new IteratorBytesSupplier(new OutputIterator(it)));
            ctx.setAttachment("out", output);
        }
        return new NDList();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public O processOutput(TranslatorContext ctx, NDList list) {
        return (O) ctx.getAttachment("out");
    }

    private TokenIterator generate(String input) {
        LlamaInput in = JsonUtils.GSON.fromJson(input, LlamaInput.class);
        return generate(in);
    }

    private TokenIterator generate(LlamaInput in) {
        InputParameters param = in.getParameters().toInputParameters();
        String prefix = in.getPrefix();
        String suffix = in.getSuffix();
        String inputs = in.getInputs();
        if (prefix != null && suffix != null) {
            LlamaLibrary.infill(handle, prefix, prefix, param);
        } else if (inputs != null && !inputs.isEmpty()) {
            LlamaLibrary.generate(handle, inputs, param);
        } else {
            throw new IllegalArgumentException("Unsupported input format");
        }
        return new TokenIterator(handle);
    }

    private static final class OutputIterator implements Iterator<BytesSupplier> {

        private TokenIterator it;

        public OutputIterator(TokenIterator it) {
            this.it = it;
        }

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return it.hasNext();
        }

        /** {@inheritDoc} */
        @Override
        public BytesSupplier next() {
            Token token = it.next();
            return BytesSupplier.wrap(JsonUtils.GSON.toJson(token) + "\n");
        }
    }
}
