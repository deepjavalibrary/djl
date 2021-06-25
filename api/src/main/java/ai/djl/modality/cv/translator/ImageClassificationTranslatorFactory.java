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

import ai.djl.Model;
import ai.djl.modality.Classifications;
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
public class ImageClassificationTranslatorFactory implements TranslatorFactory {

    private static final Set<Pair<Type, Type>> SUPPORTED_TYPES = new HashSet<>();

    static {
        SUPPORTED_TYPES.add(new Pair<>(Image.class, Classifications.class));
        SUPPORTED_TYPES.add(new Pair<>(Path.class, Classifications.class));
        SUPPORTED_TYPES.add(new Pair<>(URL.class, Classifications.class));
        SUPPORTED_TYPES.add(new Pair<>(InputStream.class, Classifications.class));
    }

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return SUPPORTED_TYPES;
    }

    /** {@inheritDoc} */
    @Override
    public Translator<?, ?> newInstance(
            Class<?> input, Class<?> output, Model model, Map<String, ?> arguments) {
        if (input == Image.class && output == Classifications.class) {
            return ImageClassificationTranslator.builder(arguments).build();
        } else if (input == Path.class && output == Classifications.class) {
            return new FileTranslator<>(ImageClassificationTranslator.builder(arguments).build());
        } else if (input == URL.class && output == Classifications.class) {
            return new UrlTranslator<>(ImageClassificationTranslator.builder(arguments).build());
        } else if (input == InputStream.class && output == Classifications.class) {
            return new InputStreamTranslator<>(
                    ImageClassificationTranslator.builder(arguments).build());
        }
        throw new IllegalArgumentException("Unsupported input/output types.");
    }
}
