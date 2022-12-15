/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray.gc;

import ai.djl.ndarray.NDArray;

/** {@code PtNDArrayProxyMaker} creates a proxy facade. */
public interface NDArrayProxyMaker {

    /**
     * Returns the size of the map.
     *
     * @return the size of the map
     */
    int mapSize();

    /**
     * Wraps the {@link NDArray} in a proxy facade.
     *
     * @param array the array to wrap
     * @return the wrapped array
     */
    NDArray wrap(NDArray array);
}
