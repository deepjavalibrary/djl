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
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;
import ai.djl.util.StringPair;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;

import java.util.ArrayList;
import java.util.List;

/** A {@link Translator} that can handle generic cross encoder {@link Input} and {@link Output}. */
public class CrossEncoderServingTranslator implements Translator<Input, Output> {

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
        ReRankingInput in = ReRankingInput.parseInput(input);
        if (in.batch != null) {
            ctx.setAttachment("batch", Boolean.TRUE);
            return translator.batchProcessInput(ctx, in.batch);
        }

        NDList ret = translator.processInput(ctx, in.pair);
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
        List<StringPair> prompts = new ArrayList<>(mapping.length);
        for (int i = 0; i < mapping.length; ++i) {
            ReRankingInput in = ReRankingInput.parseInput(inputs.get(i));
            if (in.batch != null) {
                List<StringPair> batch = in.batch;
                mapping[i] = batch.size();
                prompts.addAll(batch);
            } else {
                mapping[i] = -1;
                prompts.add(in.pair);
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

    private static final class ReRankingInput {

        private StringPair pair;
        private List<StringPair> batch;

        ReRankingInput(StringPair pair) {
            this.pair = pair;
        }

        ReRankingInput(List<StringPair> batch) {
            this.batch = batch;
        }

        static ReRankingInput parseInput(Input input) throws TranslateException {
            PairList<String, BytesSupplier> content = input.getContent();
            if (content.isEmpty()) {
                throw new TranslateException("Input data is empty.");
            }

            String contentType = input.getProperty("Content-Type", null);
            if (contentType != null) {
                int pos = contentType.indexOf(';');
                if (pos > 0) {
                    contentType = contentType.substring(0, pos);
                }
            }
            StringPair pair = null;
            if ("application/json".equals(contentType)) {
                String json = input.getData().getAsString();
                try {
                    JsonElement element = JsonUtils.GSON.fromJson(json, JsonElement.class);
                    if (element.isJsonArray()) {
                        JsonArray array = element.getAsJsonArray();
                        int size = array.size();
                        List<StringPair> batch = new ArrayList<>(size);
                        for (int i = 0; i < size; ++i) {
                            JsonObject obj = array.get(i).getAsJsonObject();
                            batch.add(parseStringPair(obj));
                        }
                        return new ReRankingInput(batch);
                    } else if (element.isJsonObject()) {
                        JsonObject obj = element.getAsJsonObject();
                        JsonElement query = obj.get("query");
                        if (query != null) {
                            String key = query.getAsString();
                            JsonArray texts = obj.get("texts").getAsJsonArray();
                            int size = texts.size();
                            List<StringPair> batch = new ArrayList<>(size);
                            for (int i = 0; i < size; ++i) {
                                String value = texts.get(i).getAsString();
                                batch.add(new StringPair(key, value));
                            }
                            return new ReRankingInput(batch);
                        } else {
                            pair = parseStringPair(obj);
                        }
                    } else {
                        throw new TranslateException("Unexpected json type");
                    }
                } catch (JsonParseException e) {
                    throw new TranslateException("Input is not a valid json.", e);
                }
            } else {
                String text = input.getAsString("text");
                String textPair = input.getAsString("text_pair");
                if (text != null && textPair != null) {
                    pair = new StringPair(text, textPair);
                }
                String key = input.getAsString("key");
                String value = input.getAsString("value");
                if (key != null && value != null) {
                    pair = new StringPair(key, value);
                }
            }

            if (pair == null) {
                throw new TranslateException("Missing key or value in input.");
            }
            return new ReRankingInput(pair);
        }

        private static StringPair parseStringPair(JsonObject json) throws TranslateException {
            JsonElement text = json.get("text");
            JsonElement textPair = json.get("text_pair");
            if (text != null && textPair != null) {
                return new StringPair(text.getAsString(), textPair.getAsString());
            }
            JsonElement key = json.get("key");
            JsonElement value = json.get("value");
            if (key != null && value != null) {
                return new StringPair(key.getAsString(), value.getAsString());
            }
            throw new TranslateException("Missing text or text_pair in json.");
        }
    }
}
