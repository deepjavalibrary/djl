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

import ai.djl.Model;
import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.JsonUtils;
import ai.djl.util.Pair;

import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/** A {@link TranslatorFactory} that creates an {@link RpcTranslator}. */
public class RpcTranslatorFactory implements TranslatorFactory {

    private TypeConverter<?, ?> converter;
    private Set<Pair<Type, Type>> supportedTypes;

    /** Constructs a {@code RpcTranslatorFactory} instance. */
    public RpcTranslatorFactory() {
        supportedTypes = Collections.emptySet();
    }

    /**
     * Constructs a {@code RpcTranslatorFactory} instance.
     *
     * @param converter the {@code TypeConverter}
     */
    public RpcTranslatorFactory(TypeConverter<?, ?> converter) {
        this.converter = converter;
        supportedTypes = Collections.singleton(converter.getSupportedType());
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments)
            throws TranslateException {
        try {
            if (!isSupported(input, output)) {
                throw new IllegalArgumentException("Unsupported input/output types.");
            }
            RpcClient client = RpcClient.getClient(arguments);
            if (converter != null) {
                return new RpcTranslator<>(client, (TypeConverter<I, O>) converter);
            }
            return new RpcTranslator<>(client, new DefaultTypeConverter<>(input, output));
        } catch (IOException e) {
            throw new TranslateException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return supportedTypes;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSupported(Class<?> input, Class<?> output) {
        if (converter == null) {
            return true;
        }
        return TranslatorFactory.super.isSupported(input, output);
    }

    private static final class DefaultTypeConverter<I, O> implements TypeConverter<I, O> {

        private Class<I> input;
        private Class<O> output;
        private Method fromJson;
        private Method fromJsonStream;

        DefaultTypeConverter(Class<I> input, Class<O> output) {
            this.input = input;
            this.output = output;
            try {
                fromJsonStream = output.getDeclaredMethod("fromJson", Iterator.class);
            } catch (ReflectiveOperationException e) {
                // ignore
            }
            try {
                fromJson = output.getDeclaredMethod("fromJson", String.class);
            } catch (ReflectiveOperationException e) {
                // ignore
            }
        }

        /** {@inheritDoc} */
        @Override
        public Pair<Type, Type> getSupportedType() {
            return new Pair<>(input, output);
        }

        /** {@inheritDoc} */
        @Override
        public Input toInput(I in) {
            if (in instanceof Input) {
                return (Input) in;
            }
            Input converted = new Input();
            converted.add(BytesSupplier.wrapAsJson(in));
            return converted;
        }

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("unchecked")
        public O fromOutput(Output out) throws TranslateException {
            if (output == Output.class) {
                return (O) out;
            }
            int code = out.getCode();
            BytesSupplier data = out.getData();
            if (code != 200) {
                String error;
                if (data == null) {
                    error = out.getMessage();
                } else {
                    error = out.getMessage() + " " + data.getAsString();
                }
                throw new TranslateException(error);
            }
            if (output == String.class) {
                return (O) data.getAsString();
            }
            try {
                if (data instanceof ChunkedBytesSupplier && fromJsonStream != null) {
                    Iterator<String> it = new ChunkIterator((ChunkedBytesSupplier) data);
                    return (O) fromJsonStream.invoke(null, it);
                } else if (fromJson != null) {
                    return (O) fromJson.invoke(null, data.getAsString());
                }
            } catch (ReflectiveOperationException e) {
                throw new TranslateException("Failed convert from json", e);
            }
            return JsonUtils.GSON.fromJson(data.getAsString(), output);
        }
    }

    private static final class ChunkIterator implements Iterator<String> {

        private ChunkedBytesSupplier cbs;
        private boolean error;

        ChunkIterator(ChunkedBytesSupplier cbs) {
            this.cbs = cbs;
        }

        @Override
        public boolean hasNext() {
            if (error) {
                return false;
            }
            return cbs.hasNext();
        }

        @Override
        public String next() {
            try {
                return new String(cbs.nextChunk(20, TimeUnit.SECONDS), StandardCharsets.UTF_8);
            } catch (InterruptedException e) {
                error = true;
                return null;
            }
        }
    }
}
