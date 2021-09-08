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
package ai.djl.modality.cv.translator.wrapper;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.InputStream;

/**
 * Built-in {@code Translator} that provides image pre-processing from {@code InputStream}.
 *
 * @param <T> the output object type
 */
public class InputStreamTranslator<T> implements Translator<InputStream, T> {

    private Translator<Image, T> translator;

    /**
     * Creates a {@code UrlTranslator} instance.
     *
     * @param translator a {@code Translator} that can process image
     */
    public InputStreamTranslator(Translator<Image, T> translator) {
        this.translator = translator;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, InputStream input) throws Exception {
        Image image = ImageFactory.getInstance().fromInputStream(input);
        return translator.processInput(ctx, image);
    }

    /** {@inheritDoc} */
    @Override
    public T processOutput(TranslatorContext ctx, NDList list) throws Exception {
        return translator.processOutput(ctx, list);
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return translator.getBatchifier();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        translator.prepare(ctx);
    }
}
