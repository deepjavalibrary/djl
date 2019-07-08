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
package software.amazon.ai;

import java.util.Collections;
import java.util.List;
import software.amazon.ai.ndarray.NDArray;

/**
 * An interface representing an initialization method.
 *
 * <p>Used to initialize the {@link NDArray} parameters stored within a {@link Block}.
 */
public interface Initializer {

    /**
     * Initializes a single {@link NDArray}.
     *
     * @param array the {@link NDArray} to initialize
     */
    default void initialize(NDArray array) {
        initialize(Collections.singletonList(array));
    }

    /**
     * Initializes a list of {@link NDArray}s.
     *
     * @param parameters the {@link NDArray}s to initialize
     */
    void initialize(List<NDArray> parameters);
}
