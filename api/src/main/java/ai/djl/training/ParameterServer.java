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

package ai.djl.training;

import ai.djl.ndarray.NDArray;
import java.util.Arrays;

/** An interface for a key-value store to store parameters, and their corresponding gradients. */
public interface ParameterServer extends AutoCloseable {

    /**
     * Initializes the {@code ParameterStore} for the given parameter.
     *
     * @param parameterId the parameter ID
     * @param value the values to be set for the given parameter
     */
    void init(String parameterId, NDArray[] value);

    /**
     * Updates the parameter of a key from Parameter Server.
     *
     * @param parameterId the key to identify the parameter
     * @param params the parameter NDArrays in different devices to be updated.
     */
    default void update(String parameterId, NDArray[] params) {
        NDArray[] grads = Arrays.stream(params).map(NDArray::getGradient).toArray(NDArray[]::new);
        update(parameterId, grads, params);
        Arrays.stream(grads).forEach(NDArray::close);
    }
    /**
     * Updates the parameter of a key from Parameter Server.
     *
     * @param parameterId the key to identify the parameter
     * @param grads the gradient NDArrays in different devices to apply the update.
     * @param params the parameter NDArrays in different devices to be updated.
     */
    void update(String parameterId, NDArray[] grads, NDArray[] params);

    /** {@inheritDoc} */
    @Override
    void close();
}
