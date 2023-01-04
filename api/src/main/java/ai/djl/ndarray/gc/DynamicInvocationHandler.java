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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/** {@code DynamicInvocationHandler} implements the {@link InvocationHandler}. */
public class DynamicInvocationHandler implements InvocationHandler {

    private static final Logger logger = LoggerFactory.getLogger(DynamicInvocationHandler.class);

    WeakHashMapWrapper<String, NDArray> map;
    String uid;

    NDArrayProxyMaker ndArrayProxyMaker;

    /**
     * Creates a new instance of {@code DynamicInvocationHandler}.
     *
     * @param uid the uid
     * @param map the map
     * @param ndArrayProxyMaker the ndArrayProxyMaker
     */
    public DynamicInvocationHandler(
            String uid,
            WeakHashMapWrapper<String, NDArray> map,
            NDArrayProxyMaker ndArrayProxyMaker) {
        this.map = map;
        this.uid = uid;
        this.ndArrayProxyMaker = ndArrayProxyMaker;
    }

    /** {@inheritDoc} */
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) {

        if ("getNumOfNDArraysInGCMap".equals(method.getName())) {
            return this.map.size();
        }
        if ("getImplementation".equals(method.getName())) {
            return map.get(uid);
        }
        Object result;
        try {
            result = method.invoke(map.get(uid), args);
        } catch (IllegalAccessException | InvocationTargetException e) {
            logger.error("Error invoking method", e);
            throw new RuntimeException(e); // NOPMD
        }

        return result;
    }
}
