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

import ai.djl.Model;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** A {@link TranslatorFactory} that creates a {@link Sam2Translator} instance. */
public class Sam2TranslatorFactory implements TranslatorFactory, Serializable {

    private static final long serialVersionUID = 1L;

    private static final Set<Pair<Type, Type>> SUPPORTED_TYPES = new HashSet<>();

    static {
        SUPPORTED_TYPES.add(new Pair<>(Sam2Input.class, DetectedObjects.class));
        SUPPORTED_TYPES.add(new Pair<>(Input.class, Output.class));
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments) {
        if (input == Sam2Input.class && output == DetectedObjects.class) {
            return (Translator<I, O>) Sam2Translator.builder(arguments).build();
        } else if (input == Input.class && output == Output.class) {
            Sam2Translator translator = Sam2Translator.builder(arguments).build();
            return (Translator<I, O>) new Sam2ServingTranslator(translator);
        }
        throw new IllegalArgumentException("Unsupported input/output types.");
    }

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return SUPPORTED_TYPES;
    }
}
