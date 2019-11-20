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
package ai.djl.translate;

import ai.djl.ndarray.NDArray;

/**
 * An interface to apply various transforms to the input.
 *
 * <p>A transform can be any function that modifies the input. Some examples of transform are crop
 * and resize.
 */
public interface Transform {

    /**
     * Applies the {@code Transform} to the given {@link NDArray}.
     *
     * @param array the {@link NDArray} on which the {@link Transform} is applied
     * @return the output of the {@code Transform}
     */
    NDArray transform(NDArray array);
}
