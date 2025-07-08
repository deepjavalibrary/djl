/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rpc;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;

/** The {@link ai.djl.translate.Translator} for model deploy on remote model server. */
public class RpcTranslator<I, O> implements NoBatchifyTranslator<I, O> {

    private RpcClient client;
    private TypeConverter<I, O> converter;

    /**
     * Constructs a {@code RpcTranslator} instance.
     *
     * @param client a {@link RpcClient} connects to remote model server
     * @param converter a {@link TypeConverter} to convert data type
     */
    protected RpcTranslator(RpcClient client, TypeConverter<I, O> converter) {
        this.client = client;
        this.converter = converter;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, I input) throws IOException {
        Input in = converter.toInput(input);
        Output output = client.send(in);
        ctx.setAttachment("output", output);
        return new NDList();
    }

    /** {@inheritDoc} */
    @Override
    public O processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
        Output output = (Output) ctx.getAttachment("output");
        return converter.fromOutput(output);
    }
}
