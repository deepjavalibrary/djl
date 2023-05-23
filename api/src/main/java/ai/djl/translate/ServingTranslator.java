/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.inference.streaming.IteratorBytesSupplier;
import ai.djl.inference.streaming.PublisherBytesSupplier;
import ai.djl.inference.streaming.StreamingTranslator;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;

import java.util.Iterator;
import java.util.Map;
import java.util.stream.Stream;

/** A {@link Translator} that can handle generic {@link Input} and {@link Output}. */
public interface ServingTranslator extends StreamingTranslator<Input, Output> {

    /**
     * Sets the configurations for the {@code Translator} instance.
     *
     * @param arguments the configurations for the {@code Translator} instance
     */
    void setArguments(Map<String, ?> arguments);

    /** {@inheritDoc} */
    @Override
    default Support getSupport() {
        return Support.BOTH;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.AvoidThrowingRawExceptionTypes")
    default StreamOutput<Output> processStreamOutput(TranslatorContext ctx, Stream<NDList> list) {
        return new StreamOutput<Output>() {

            @Override
            protected Output buildAsyncOutput() {
                PublisherBytesSupplier bytesSupplier = new PublisherBytesSupplier();
                Output output = new Output();
                output.add(bytesSupplier);
                return output;
            }

            @Override
            protected void computeAsyncOutputInternal(Output output) {
                PublisherBytesSupplier bytesSupplier = (PublisherBytesSupplier) output.getData();
                Iterator<NDList> it = list.iterator();
                while (it.hasNext()) {
                    try {
                        bytesSupplier.appendContent(
                                processOutput(ctx, it.next()).getData().getAsBytes(),
                                !it.hasNext());
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            }

            @Override
            public Output getIterativeOutputInternal() {
                Iterator<BytesSupplier> outputs =
                        list.map(
                                        ndList -> {
                                            try {
                                                return processOutput(ctx, ndList).getData();
                                            } catch (Exception e) {
                                                throw new RuntimeException(e);
                                            }
                                        })
                                .iterator();
                IteratorBytesSupplier bytesSupplier = new IteratorBytesSupplier(outputs);
                Output output = new Output();
                output.add(bytesSupplier);
                return output;
            }
        };
    }
}
