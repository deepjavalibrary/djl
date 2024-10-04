/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.translator;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;

/** A {@link Translator} that can serve SAM2 model. */
public class Sam2ServingTranslator implements Translator<Input, Output> {

    private Sam2Translator translator;

    /**
     * Constructs a new {@code Sam2ServingTranslator} instance.
     *
     * @param translator a {@code Sam2Translator}
     */
    public Sam2ServingTranslator(Sam2Translator translator) {
        this.translator = translator;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return translator.getBatchifier();
    }

    /** {@inheritDoc} */
    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) throws Exception {
        Output output = new Output();
        output.addProperty("Content-Type", "application/json");
        DetectedObjects obj = translator.processOutput(ctx, list);
        output.add(BytesSupplier.wrapAsJson(obj));
        return output;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
        BytesSupplier data = input.getData();
        try {
            if (data == null) {
                throw new TranslateException("Input data is empty.");
            }
            Sam2Input sam2 = Sam2Input.fromJson(data.getAsString());
            return translator.processInput(ctx, sam2);
        } catch (IOException e) {
            throw new TranslateException("Input is not an Image data type", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        translator.prepare(ctx);
    }
}
