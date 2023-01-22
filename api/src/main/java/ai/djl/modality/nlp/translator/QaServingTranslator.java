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
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;

import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;

/**
 * A {@link Translator} that can handle generic question answering {@link Input} and {@link Output}.
 */
public class QaServingTranslator implements NoBatchifyTranslator<Input, Output> {

    private Translator<QAInput, String> translator;
    private Translator<QAInput[], String[]> batchTranslator;

    /**
     * Constructs a {@code QaServingTranslator} instance.
     *
     * @param translator a {@code Translator} processes question answering input
     */
    public QaServingTranslator(Translator<QAInput, String> translator) {
        this.translator = translator;
        this.batchTranslator = translator.toBatchTranslator();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        translator.prepare(ctx);
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
        QAInput qa;
        if ("application/json".equals(contentType)) {
            String json = input.getData().getAsString();
            try {
                JsonElement element = JsonUtils.GSON.fromJson(json, JsonElement.class);
                if (element.isJsonArray()) {
                    ctx.setAttachment("batch", Boolean.TRUE);
                    QAInput[] inputs = JsonUtils.GSON.fromJson(json, QAInput[].class);
                    return batchTranslator.processInput(ctx, inputs);
                }

                qa = JsonUtils.GSON.fromJson(json, QAInput.class);
            } catch (JsonParseException e) {
                throw new TranslateException("Input is not a valid json.", e);
            }
        } else if (content.contains("question") && content.contains("paragraph")) {
            String question = input.getAsString("question");
            String paragraph = input.getAsString("paragraph");
            qa = new QAInput(question, paragraph);
        } else {
            throw new TranslateException("Not a QuestionAnswering input.");
        }

        NDList ret = translator.processInput(ctx, qa);
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
            output.add(translator.processOutput(ctx, list));
        }
        return output;
    }
}
