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
import ai.djl.translate.PreProcessor;
import ai.djl.translate.TranslatorContext;

import java.net.URL;

/**
 * Built-in {@code PreProcessor} that provides image pre-processing from URL.
 *
 * @param <T> the output object type
 */
public class UrlImagePreProcessor<T> implements PreProcessor<URL> {

    private PreProcessor<Image> preProcessor;

    /**
     * Creates a {@code UrlImagePreProcessor} instance.
     *
     * @param preProcessor a {@code PreProcessor} that can process image
     */
    public UrlImagePreProcessor(PreProcessor<Image> preProcessor) {
        this.preProcessor = preProcessor;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, URL input) throws Exception {
        Image image = ImageFactory.getInstance().fromUrl(input);
        return preProcessor.processInput(ctx, image);
    }
}
