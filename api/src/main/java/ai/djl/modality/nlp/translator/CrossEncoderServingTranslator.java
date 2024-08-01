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
package ai.djl.modality.nlp.translator;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;
import ai.djl.util.StringPair;

import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;

/** A {@link Translator} that can handle generic cross encoder {@link Input} and {@link Output}. */
public class CrossEncoderServingTranslator implements NoBatchifyTranslator<Input, Output> {

    private static final Type LIST_TYPE = new TypeToken<List<StringPair>>() {}.getType();

    private Translator<StringPair, float[]> translator;

    /**
     * Constructs a {@code CrossEncoderServingTranslator} instance.
     *
     * @param translator a {@code Translator} processes question answering input
     */
    public CrossEncoderServingTranslator(Translator<StringPair, float[]> translator) {
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
        PairList<String, BytesSupplier> content = input.getContent();
        if (content.isEmpty()) {
            throw new TranslateException("Input data is empty.");
        }

        String contentType = input.getProperty("Content-Type", null);
        StringPair pair;
        if ("application/json".equals(contentType)) {
            String json = input.getData().getAsString();
            try {
                JsonElement element = JsonUtils.GSON.fromJson(json, JsonElement.class);
                if (element.isJsonArray()) {
                    ctx.setAttachment("batch", Boolean.TRUE);
                    List<StringPair> inputs = JsonUtils.GSON.fromJson(json, LIST_TYPE);
                    return translator.batchProcessInput(ctx, inputs);
                }

                pair = JsonUtils.GSON.fromJson(json, StringPair.class);
                if (pair.getKey() == null || pair.getValue() == null) {
                    throw new TranslateException("Missing key or value in json.");
                }
            } catch (JsonParseException e) {
                throw new TranslateException("Input is not a valid json.", e);
            }
        } else {
            String key = input.getAsString("key");
            String value = input.getAsString("value");
            if (key == null || value == null) {
                throw new TranslateException("Missing key or value in input.");
            }
            pair = new StringPair(key, value);
        }

        NDList ret = translator.processInput(ctx, pair);
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
}
