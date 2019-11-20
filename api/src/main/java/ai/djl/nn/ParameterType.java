/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.nn;

import ai.djl.training.initializer.Initializer;

/** Enumerates the types of {@link Parameter}. */
public enum ParameterType {
    WEIGHT(null),
    BIAS(Initializer.ZEROS),
    GAMMA(Initializer.ONES),
    BETA(Initializer.ZEROS),
    RUNNING_MEAN(Initializer.ZEROS),
    RUNNING_VAR(Initializer.ONES),
    OTHER(null);

    private final transient Initializer initializer;

    ParameterType(Initializer initializer) {
        this.initializer = initializer;
    }

    /**
     * Gets the {@link Initializer} of this {@code ParameterType}.
     *
     * @return the {@link Initializer} of this {@code ParameterType}
     */
    public Initializer getInitializer() {
        return initializer;
    }
}
