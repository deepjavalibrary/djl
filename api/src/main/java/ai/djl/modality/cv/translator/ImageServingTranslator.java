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
package ai.djl.modality.cv.translator;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonSerializable;
import java.io.ByteArrayInputStream;
import java.io.IOException;

/** A {@link Translator} that can handle generic CV {@link Input} and {@link Output}. */
public class ImageServingTranslator implements Translator<Input, Output> {

    private Translator<Image, ?> translator;
    private ImageFactory factory;

    /**
     * Constructs a new {@code ImageServingTranslator} instance.
     *
     * @param translator a {@code Translator} processes Image input
     */
    public ImageServingTranslator(Translator<Image, ?> translator) {
        this.translator = translator;
        factory = ImageFactory.getInstance();
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
        Object obj = translator.processOutput(ctx, list);
        if (obj instanceof JsonSerializable) {
            output.add((JsonSerializable) obj);
        } else {
            output.add(BytesSupplier.wrapAsJson(obj));
        }
        output.addProperty("Content-Type", "application/json");
        return output;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
        BytesSupplier data = input.getData();
        try {
            Image image = factory.fromInputStream(new ByteArrayInputStream(data.getAsBytes()));
            return translator.processInput(ctx, image);
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
