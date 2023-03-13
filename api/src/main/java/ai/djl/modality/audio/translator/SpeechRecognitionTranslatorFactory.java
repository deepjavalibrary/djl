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
package ai.djl.modality.audio.translator;

import ai.djl.Model;
import ai.djl.modality.audio.Audio;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** A {@link TranslatorFactory} that creates a {@link SpeechRecognitionTranslator} instance. */
public class SpeechRecognitionTranslatorFactory implements TranslatorFactory, Serializable {

    private static final long serialVersionUID = 1L;

    private static final Set<Pair<Type, Type>> SUPPORTED_TYPES = new HashSet<>();

    static {
        SUPPORTED_TYPES.add(new Pair<>(Audio.class, String.class));
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
        if (input == Audio.class && output == String.class) {
            return (Translator<I, O>) new SpeechRecognitionTranslator();
        }
        throw new IllegalArgumentException("Unsupported input/output types.");
    }
}
