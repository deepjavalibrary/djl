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

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.translator.wrapper.FileImagePreProcessor;
import ai.djl.modality.cv.translator.wrapper.InputStreamImagePreProcessor;
import ai.djl.modality.cv.translator.wrapper.StringImagePreProcessor;
import ai.djl.modality.cv.translator.wrapper.UrlImagePreProcessor;
import ai.djl.translate.ExpansionTranslatorFactory;
import ai.djl.translate.PreProcessor;
import ai.djl.util.Pair;

import java.io.InputStream;
import java.lang.reflect.Type;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * A helper to create a {@link ai.djl.translate.TranslatorFactory} with the {@link
 * BaseImageTranslator}.
 *
 * @param <O> the output type for the {@link ai.djl.translate.TranslatorFactory}.
 */
public abstract class BaseImageTranslatorFactory<O> extends ExpansionTranslatorFactory<Image, O> {

    /** {@inheritDoc} */
    @Override
    protected Map<Type, Function<PreProcessor<Image>, PreProcessor<?>>>
            getPreprocessorExpansions() {
        Map<Type, Function<PreProcessor<Image>, PreProcessor<?>>> expansions =
                new ConcurrentHashMap<>();
        expansions.put(Path.class, FileImagePreProcessor::new);
        expansions.put(URL.class, UrlImagePreProcessor::new);
        expansions.put(String.class, StringImagePreProcessor::new);
        expansions.put(InputStream.class, InputStreamImagePreProcessor::new);
        return expansions;
    }

    /** {@inheritDoc} */
    @Override
    protected Map<Pair<Type, Type>, TranslatorExpansion<Image, O>> getExpansions() {
        Map<Pair<Type, Type>, TranslatorExpansion<Image, O>> expansions = new ConcurrentHashMap<>();
        expansions.put(new Pair<>(Input.class, Output.class), ImageServingTranslator::new);
        return expansions;
    }

    /** {@inheritDoc} */
    @Override
    public Class<Image> getBaseInputType() {
        return Image.class;
    }
}
