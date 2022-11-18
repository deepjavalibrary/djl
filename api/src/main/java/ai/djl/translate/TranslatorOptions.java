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
package ai.djl.translate;

import ai.djl.util.Pair;

import java.lang.reflect.Type;
import java.util.Set;

/** A set of possible options for {@link Translator}s with different input and output types. */
public interface TranslatorOptions {

    /**
     * Returns the supported wrap types.
     *
     * @return the supported wrap types
     * @see #option(Class, Class)
     */
    Set<Pair<Type, Type>> getOptions();

    /**
     * Returns if the input/output is a supported wrap type.
     *
     * @param input the input class
     * @param output the output class
     * @return {@code true} if the input/output type is supported
     * @see #option(Class, Class)
     */
    default boolean isSupported(Class<?> input, Class<?> output) {
        return getOptions().contains(new Pair<Type, Type>(input, output));
    }

    /**
     * Returns the {@link Translator} option with the matching input and output type.
     *
     * @param <I> the input data type
     * @param <O> the output data type
     * @param input the input class
     * @param output the output class
     * @return a new instance of the {@code Translator} class
     * @throws TranslateException if failed to create Translator instance
     */
    <I, O> Translator<I, O> option(Class<I> input, Class<O> output) throws TranslateException;
}
