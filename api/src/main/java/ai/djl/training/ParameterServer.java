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

public interface ParameterServer extends AutoCloseable {

    void init(String parameterId, NDArray[] value);

    /**
     * Push values into key in Parameter Server.
     *
     * @param parameterId key to update
     * @param grads values corresponding to key, values in array will be summed when update key
     * @param priority The priority of the push operation. Higher priority push operations are
     *     likely to be executed before other push actions
     */
    void push(String parameterId, NDArray[] grads, int priority);

    /**
     * Pull value of key from Parameter Server to NDArrays.
     *
     * @param parameterId key to pull
     * @param weights NDArrays to store the value corresponding to key, value will be copied to the
     *     devices of the NDArrays
     * @param priority The priority of the push operation. Higher priority push operations are
     *     likely to be executed before other push actions
     */
    void pull(String parameterId, NDArray[] weights, int priority);

    @Override
    void close();
}
