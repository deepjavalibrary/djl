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
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import java.net.URL;
import java.util.Map;

/**
 * A factory class creates {@link Translator} that can process image from a URL.
 *
 * @param <T> the type of the output
 */
public class UrlTranslatorFactory<T> implements TranslatorFactory<URL, T> {

    private TranslatorFactory<Image, T> factory;

    /**
     * Creates a {@code FileTranslatorFactory} instance.
     *
     * @param factory a factory that can process image
     */
    public UrlTranslatorFactory(TranslatorFactory<Image, T> factory) {
        this.factory = factory;
    }

    /** {@inheritDoc} */
    @Override
    public Translator<URL, T> newInstance(Map<String, Object> arguments) {
        return new UrlTranslator<>(factory.newInstance(arguments));
    }
}
