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
package ai.djl.nn.transformer;

import ai.djl.ndarray.LazyNDArray;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

/**
 * Helper class for more complicated memory management scenarios. Allows to avoid boilerplate for
 * memory handling. Makes sure the sub NDManager used is connected to the correct GPU to avoid
 * crashes.
 */
public final class MemoryScope implements AutoCloseable {

    private final NDManager parentManager;
    private final NDManager subManager;

    private MemoryScope(NDManager parentManager, NDManager subManager) {
        this.parentManager = parentManager;
        this.subManager = subManager;
    }

    /**
     * Adds all arrays in the given lists to this memory scope.
     *
     * @param lists the lists whose arrays to add to this scope, may be empty
     * @return this scope
     */
    public MemoryScope add(NDList... lists) {
        for (NDList list : lists) {
            list.attach(subManager);
        }
        return this;
    }

    /**
     * Adds the given arrays to this scopes sub manager.
     *
     * @param arrays the arrays to add
     * @return this scope
     */
    public MemoryScope add(NDArray... arrays) {
        for (NDArray array : arrays) {
            array.attach(subManager);
        }
        return this;
    }

    /**
     * Remove the given arrays from this scope and attach them back to this scopes parent NDManager.
     *
     * @param lists the lists containing the arrays to remove
     * @return this scope
     */
    public MemoryScope remove(NDList... lists) {
        for (NDList list : lists) {
            list.attach(parentManager);
        }
        return this;
    }

    /**
     * Remove the given arrays from this scope and attach them back to this scopes parent NDManager.
     *
     * @param arrays arrays to remove
     * @return this scope
     */
    public MemoryScope remove(NDArray... arrays) {
        for (NDArray array : arrays) {
            array.attach(parentManager);
        }
        return this;
    }

    /**
     * Returns the NDManager used to manage this scopes resources.
     *
     * @return the NDManager used to manage this scopes resources
     */
    public NDManager getScopeManager() {
        return subManager;
    }

    /**
     * Waits for all given arrays to be ready to read, i.e. waits for pending computations that
     * write to them, then removes them from this scope.
     *
     * @param arrays arrays to wait for
     * @return this scope
     */
    public MemoryScope waitToRead(NDArray... arrays) {
        for (NDArray array : arrays) {
            if (array instanceof LazyNDArray) {
                LazyNDArray lazyNDArray = (LazyNDArray) array;
                lazyNDArray.waitToRead();
            }
            remove(array);
        }
        return this;
    }

    /**
     * Waits for all arrays in all given lists to be ready to be read, i.e. waits for pending
     * computations that write to them, then removes them from this scope.
     *
     * @param lists may be empty
     * @return this scope
     */
    public MemoryScope waitToRead(NDList... lists) {
        for (NDList list : lists) {
            if (list != null) {
                for (NDArray array : list) {
                    waitToRead(array);
                }
            }
        }
        return this;
    }

    /**
     * Closes this scope by closing the sub manager used to manage it. This causes all arrays still
     * attached to this scope to be closed as well.
     */
    @Override
    public void close() {
        subManager.close();
    }

    /**
     * Creates a new memory scope for the device of the given array and adds the array.
     *
     * @param ndArray an array
     * @return a new memory scrope containing the array
     */
    public static MemoryScope from(final NDArray ndArray) {
        return new MemoryScope(
                        ndArray.getManager(),
                        ndArray.getManager().newSubManager(ndArray.getDevice()))
                .add(ndArray);
    }

    /**
     * Creates a new memory scope that fits the device of the first array in the given list, adds
     * all arrays in the given list.
     *
     * @param list a list of arrays, must not be empty
     * @return a new memory scope
     */
    public static MemoryScope from(final NDList list) {
        final NDArray ndArray = list.head();
        return new MemoryScope(
                        ndArray.getManager(),
                        ndArray.getManager().newSubManager(ndArray.getDevice()))
                .add(list);
    }
}
