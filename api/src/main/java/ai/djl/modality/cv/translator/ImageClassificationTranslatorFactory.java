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
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.translator.wrapper.FileTranslator;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslator;
import ai.djl.modality.cv.translator.wrapper.UrlTranslator;
import ai.djl.translate.ExpansionTranslatorFactory;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;

import java.io.InputStream;
import java.lang.reflect.Type;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/** A {@link TranslatorFactory} that creates an {@link ImageClassificationTranslator}. */
public class ImageClassificationTranslatorFactory
        extends ExpansionTranslatorFactory<Image, Classifications> {

    private static final Map<
                    Pair<Type, Type>,
                    Function<Translator<Image, Classifications>, Translator<?, ?>>>
            EXPANSIONS = new ConcurrentHashMap<>();

    static {
        EXPANSIONS.put(new Pair<>(Image.class, Classifications.class), t -> t);
        EXPANSIONS.put(new Pair<>(Path.class, Classifications.class), FileTranslator::new);
        EXPANSIONS.put(new Pair<>(URL.class, Classifications.class), UrlTranslator::new);
        EXPANSIONS.put(
                new Pair<>(InputStream.class, Classifications.class), InputStreamTranslator::new);
        EXPANSIONS.put(new Pair<>(Input.class, Output.class), ImageServingTranslator::new);
    }

    /** {@inheritDoc} */
    @Override
    protected Translator<Image, Classifications> buildBaseTranslator(
            Model model, Map<String, ?> arguments) {
        return ImageClassificationTranslator.builder(arguments).build();
    }

    /** {@inheritDoc} */
    @Override
    protected Map<Pair<Type, Type>, Function<Translator<Image, Classifications>, Translator<?, ?>>>
            getExpansions() {
        return EXPANSIONS;
    }
}
