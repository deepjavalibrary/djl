/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.mxnet.jna;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * A generic object pool implementation.
 *
 * @param <T> the type of object to put in the pool
 */
@SuppressWarnings("MissingJavadocMethod")
public class ObjectPool<T> {

    private Queue<T> queue;
    private Supplier<T> supplier;
    private Consumer<T> consumer;

    public ObjectPool(Supplier<T> supplier, Consumer<T> consumer) {
        queue = new ConcurrentLinkedQueue<>();
        this.supplier = supplier;
        this.consumer = consumer;
    }

    public T acquire() {
        T item = queue.poll();
        if (item == null) {
            if (supplier != null) {
                return supplier.get();
            }
        }
        return item;
    }

    public void recycle(T item) {
        if (consumer != null) {
            consumer.accept(item);
        }
        queue.add(item);
    }
}
