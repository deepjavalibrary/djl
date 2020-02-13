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

import java.util.Map;

/**
 * A utility class creates {@link Translator} instances.
 *
 * @param <I> the type of the input
 * @param <O> the type of the output
 */
public interface TranslatorFactory<I, O> {

    /**
     * Returns a new instance of the {@link Translator} class.
     *
     * @param arguments the configurations for a new {@code Translator} instance
     * @return a new instance of the {@code Translator} class
     */
    Translator<I, O> newInstance(Map<String, Object> arguments);
}
