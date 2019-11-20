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
     * Updates values of a key in Parameter Server.
     *
     * @param parameterId the key to update
     * @param grads the values corresponding to the key, values in array will be summed when key is
     *     updated
     * @param priority the priority of the push operation. Higher priority push operations are
     *     likely to be executed before other push actions
     */
    void push(String parameterId, NDArray[] grads, int priority);

    /**
     * Pulls the value of a key from Parameter Server to NDArrays.
     *
     * @param parameterId the key to pull
     * @param weights the NDArrays to store the value corresponding to the key, value will be copied
     *     to the devices of the NDArrays
     * @param priority the priority of the push operation. Higher priority push operations are
     *     likely to be executed before other push actions
     */
    void pull(String parameterId, NDArray[] weights, int priority);

    /** {@inheritDoc} */
    @Override
    void close();
}
