/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.IdentityHashMap;

/**
 * A class that tracks {@link NDResource} objects created in the try-with-resource block and close
 * them automatically when out of the block scope.
 *
 * <p>This class has been derived from {@code org.bytedeco.javacpp.PointerScope} by Samuel Audet
 */
public class NDScope implements AutoCloseable {

    private static final ThreadLocal<Deque<NDScope>> SCOPE_STACK =
            ThreadLocal.withInitial(ArrayDeque::new);

    private IdentityHashMap<NDArray, NDArray> resources;

    /** Constructs a new {@code NDScope} instance. */
    public NDScope() {
        resources = new IdentityHashMap<>();
        SCOPE_STACK.get().addLast(this);
    }

    /**
     * Registers {@link NDArray} object to this scope.
     *
     * @param array the {@link NDArray} object
     */
    public static void register(NDArray array) {
        Deque<NDScope> queue = SCOPE_STACK.get();
        if (queue.isEmpty()) {
            return;
        }
        queue.getLast().resources.put(array, array);
    }

    /**
     * Unregisters {@link NDArray} object from this scope.
     *
     * @param array the {@link NDArray} object
     */
    public static void unregister(NDArray array) {
        Deque<NDScope> queue = SCOPE_STACK.get();
        if (queue.isEmpty()) {
            return;
        }
        queue.getLast().resources.remove(array);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        for (NDArray array : resources.keySet()) {
            array.close();
        }
        SCOPE_STACK.get().remove(this);
    }

    /**
     * A method that does nothing.
     *
     * <p>You may use it if you do not have a better way to suppress the warning of a created but
     * not explicitly used scope.
     */
    public void suppressNotUsedWarning() {
        // do nothing
    }
}
