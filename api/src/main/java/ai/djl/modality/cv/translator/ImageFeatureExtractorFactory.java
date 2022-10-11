/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Model;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.translator.wrapper.FileTranslator;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslator;
import ai.djl.modality.cv.translator.wrapper.UrlTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;

import java.io.InputStream;
import java.lang.reflect.Type;
import java.net.URL;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** A {@link TranslatorFactory} that creates an {@link ImageClassificationTranslator}. */
public class ImageFeatureExtractorFactory implements TranslatorFactory {

    private static final Set<Pair<Type, Type>> SUPPORTED_TYPES = new HashSet<>();

    static {
        SUPPORTED_TYPES.add(new Pair<>(Image.class, byte[].class));
        SUPPORTED_TYPES.add(new Pair<>(Path.class, byte[].class));
        SUPPORTED_TYPES.add(new Pair<>(URL.class, byte[].class));
        SUPPORTED_TYPES.add(new Pair<>(InputStream.class, byte[].class));
        SUPPORTED_TYPES.add(new Pair<>(Input.class, Output.class));
    }

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return SUPPORTED_TYPES;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments) {
        ImageFeatureExtractor translator = ImageFeatureExtractor.builder(arguments).build();
        if (input == Image.class && output == byte[].class) {
            return (Translator<I, O>) translator;
        } else if (input == Path.class && output == byte[].class) {
            return (Translator<I, O>) new FileTranslator<>(translator);
        } else if (input == URL.class && output == byte[].class) {
            return (Translator<I, O>) new UrlTranslator<>(translator);
        } else if (input == InputStream.class && output == byte[].class) {
            return (Translator<I, O>) new InputStreamTranslator<>(translator);
        } else if (input == Input.class && output == Output.class) {
            return (Translator<I, O>) new ImageServingTranslator(translator);
        }
        throw new IllegalArgumentException("Unsupported input/output types.");
    }
}
