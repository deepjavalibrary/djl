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

/**
 * The gradient collector should be opened with a try-with-resources. First call collectFor on
 * whatever data you want to collect gradients for, then execute the operations. Call
 * collectProgress to execute backwards multiple times and collect to execute backwards for the
 * final time.
 */
public interface GradientCollector extends AutoCloseable {

    /**
     * Run backward and calculate gradient w.r.t previously marked variable (head).
     *
     * @param target target NDArray to run backward and calculate gradient w.r.t head
     */
    void backward(NDArray target);

    /** {@inheritDoc} */
    @Override
    void close();
}
