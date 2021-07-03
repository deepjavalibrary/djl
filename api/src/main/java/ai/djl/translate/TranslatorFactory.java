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
package ai.djl.translate;

import ai.djl.Model;
import ai.djl.util.Pair;
import java.lang.reflect.Type;
import java.util.Map;
import java.util.Set;

/** A utility class creates {@link Translator} instances. */
public interface TranslatorFactory {

    /**
     * Returns supported input/output classes.
     *
     * @return a set of supported input/output classes
     */
    Set<Pair<Type, Type>> getSupportedTypes();

    /**
     * Returns if the input/output is supported by the {@code TranslatorFactory}.
     *
     * @param input the input class
     * @param output the output class
     * @return {@code true} if the input/output type is supported
     */
    default boolean isSupported(Class<?> input, Class<?> output) {
        return getSupportedTypes().contains(new Pair<Type, Type>(input, output));
    }

    /**
     * Returns a new instance of the {@link Translator} class.
     *
     * @param input the input class
     * @param output the output class
     * @param model the {@link Model} that uses the {@link Translator}
     * @param arguments the configurations for a new {@code Translator} instance
     * @return a new instance of the {@code Translator} class
     * @throws TranslateException if failed to create Translator instance
     */
    Translator<?, ?> newInstance(
            Class<?> input, Class<?> output, Model model, Map<String, ?> arguments)
            throws TranslateException;
}
