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
package ai.djl.pytorch.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.gc.DynamicInvocationHandler;
import ai.djl.ndarray.gc.NDArrayProxyMaker;
import ai.djl.ndarray.gc.WeakHashMapWrapper;

import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicLong;

/** {@code PtNDArrayProxyMaker} creates a proxy facade. */
public class PtNDArrayProxyMaker implements NDArrayProxyMaker {

    WeakHashMapWrapper<String, NDArray> map = new WeakHashMapWrapper<>();

    AtomicLong counter = new AtomicLong(0);

    /** {@inheritDoc} */
    @Override
    public int mapSize() {
        return map.size();
    }

    /**
     * Wraps the {@link PtNDArray} in a proxy facade.
     *
     * @param array the array to wrap
     * @return the wrapped array
     */
    @Override
    public PtNDArray wrap(NDArray array) {
        String uid = array.getUid() + "-" + counter.incrementAndGet();
        map.put(uid, array);
        DynamicInvocationHandler handler = new DynamicInvocationHandler(uid, map, this);
        return (PtNDArray)
                Proxy.newProxyInstance(
                        Thread.currentThread().getContextClassLoader(),
                        new Class<?>[] {PtNDArray.class},
                        handler);
    }
}
