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
package ai.djl.translate;

import ai.djl.Model;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;

import java.lang.reflect.Type;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

/** A {@link TranslatorFactory} that creates a {@code RawTranslator} instance. */
public class NoopServingTranslatorFactory implements TranslatorFactory {

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return Collections.singleton(new Pair<>(Input.class, Output.class));
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments) {
        if (!isSupported(input, output)) {
            throw new IllegalArgumentException("Unsupported input/output types.");
        }
        String batchifier = ArgumentsUtil.stringValue(arguments, "batchifier", "none");
        return (Translator<I, O>) new NoopServingTranslator(Batchifier.fromString(batchifier));
    }

    static final class NoopServingTranslator implements Translator<Input, Output> {

        private Batchifier batchifier;

        NoopServingTranslator(Batchifier batchifier) {
            this.batchifier = batchifier;
        }

        /** {@inheritDoc} */
        @Override
        public Batchifier getBatchifier() {
            return batchifier;
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Input input) throws TranslateException {
            NDManager manager = ctx.getNDManager();
            try {
                ctx.setAttachment("properties", input.getProperties());
                return input.getDataAsNDList(manager);
            } catch (IllegalArgumentException e) {
                throw new TranslateException("Input is not a NDList data type", e);
            }
        }

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("unchecked")
        public Output processOutput(TranslatorContext ctx, NDList list) {
            Map<String, String> prop = (Map<String, String>) ctx.getAttachment("properties");
            String contentType = prop.get("Content-Type");
            String accept = prop.get("Accept");

            Output output = new Output();
            if ("tensor/npz".equalsIgnoreCase(accept)
                    || "tensor/npz".equalsIgnoreCase(contentType)) {
                output.add(list.encode(true));
                output.addProperty("Content-Type", "tensor/npz");
            } else {
                output.add(list.encode(false));
                output.addProperty("Content-Type", "tensor/ndlist");
            }
            return output;
        }
    }
}
