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
import ai.djl.ndarray.NDManager;

/**
 * Represents a supplier of {@link NDArray}.
 *
 * <p>There is no requirement that a new or distinct result be returned each time the supplier is
 * invoked.
 *
 * <p>This is a functional interface whose functional method is {@link #get(NDManager)}.
 */
@FunctionalInterface
public interface NDArraySupplier {

    /**
     * Gets an {@link NDArray} from the given {@link NDManager}.
     *
     * @param manager the {@link NDManager}
     * @return a result {@link NDArray}
     */
    NDArray get(NDManager manager);
}
