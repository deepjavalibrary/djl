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
package ai.djl.translate;

import ai.djl.Model;
import ai.djl.modality.cv.translator.ImageClassificationTranslatorFactory;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;
import java.lang.reflect.Type;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** A default implementation of {@link TranslatorFactory}. */
public class DefaultTranslatorFactory implements TranslatorFactory {

    protected Map<Pair<Type, Type>, Translator<?, ?>> translators;
    private ServingTranslatorFactory servingTranslatorFactory = new ServingTranslatorFactory();
    private ImageClassificationTranslatorFactory imageClassificationTranslatorFactory =
            new ImageClassificationTranslatorFactory();

    /**
     * Registers a {@link Translator} with the {@code TranslatorFactory}.
     *
     * @param input the input data type
     * @param output the output data type
     * @param translator the {@code Translator} to be registered
     * @param <I> the model input type
     * @param <O> the model output type
     */
    public <I, O> void registerTranslator(
            Class<I> input, Class<O> output, Translator<I, O> translator) {
        if (translators == null) {
            translators = new ConcurrentHashMap<>();
        }
        translators.put(new Pair<>(input, output), translator);
    }

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        Set<Pair<Type, Type>> set = new HashSet<>();
        if (translators != null) {
            set.addAll(translators.keySet());
        }
        set.add(new Pair<>(NDList.class, NDList.class));
        return set;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSupported(Class<?> input, Class<?> output) {
        if (input == NDList.class && output == NDList.class) {
            return true;
        }
        if (translators != null && translators.containsKey(new Pair<Type, Type>(input, output))) {
            return true;
        }
        return servingTranslatorFactory.isSupported(input, output)
                || imageClassificationTranslatorFactory.isSupported(input, output);
    }

    /** {@inheritDoc} */
    @Override
    public Translator<?, ?> newInstance(
            Class<?> input, Class<?> output, Model model, Map<String, ?> arguments)
            throws TranslateException {
        if (translators != null) {
            Translator<?, ?> translator = translators.get(new Pair<Type, Type>(input, output));
            if (translator != null) {
                return translator;
            }
        }

        if (input == NDList.class && output == NDList.class) {
            return new NoopTranslator();
        }
        if (servingTranslatorFactory.isSupported(input, output)) {
            return servingTranslatorFactory.newInstance(input, output, model, arguments);
        }
        if (imageClassificationTranslatorFactory.isSupported(input, output)) {
            return imageClassificationTranslatorFactory.newInstance(
                    input, output, model, arguments);
        }
        return null;
    }
}
