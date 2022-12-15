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

import java.lang.reflect.Proxy;
import java.util.UUID;

/** {@code NDArrayWrapFactory} creates a proxy facade. */
public class NDArrayWrapFactory {

    WeakHashMapWrapper<UUID, NDArray> map = new WeakHashMapWrapper<>();

    /**
     * Returns the size of the map.
     *
     * @return the size of the map
     */
    public int mapSize() {
        return map.size();
    }

    /**
     * Wraps the {@link NDArray} in a proxy facade.
     *
     * @param array the array to wrap
     * @return the wrapped array
     */
    public NDArray wrap(NDArray array) {
        UUID uuid = UUID.randomUUID();
        map.put(uuid, array);

        DynamicInvocationHandler handler = new DynamicInvocationHandler(uuid, map, this);
        return (NDArray)
                Proxy.newProxyInstance(
                        Thread.currentThread().getContextClassLoader(),
                        new Class<?>[] {NDArray.class},
                        handler);
    }
}
