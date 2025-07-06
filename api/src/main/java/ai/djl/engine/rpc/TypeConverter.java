/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rpc;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;

import java.lang.reflect.Type;

/**
 * A {code TypeConverter} interface defines how data type is converted to Input/Output.
 *
 * @param <I> the target input type
 * @param <O> the target output type
 */
public interface TypeConverter<I, O> {

    /**
     * Returns the supported type.
     *
     * @return the supported type
     */
    Pair<Type, Type> getSupportedType();

    /**
     * Convert the data type to {@link Input}.
     *
     * @param in the input data
     * @return the converted data
     */
    Input toInput(I in);

    /**
     * Convert the {@link Output} to target data type.
     *
     * @param out the output data
     * @return the converted data
     */
    O fromOutput(Output out) throws TranslateException;
}
