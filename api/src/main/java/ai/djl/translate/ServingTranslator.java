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

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import java.util.Map;

/** A {@link Translator} that can handle generic {@link Input} and {@link Output}. */
public interface ServingTranslator extends Translator<Input, Output> {

    /**
     * Sets the configurations for the {@code Translator} instance.
     *
     * @param arguments the configurations for the {@code Translator} instance
     */
    void setArguments(Map<String, ?> arguments);
}
