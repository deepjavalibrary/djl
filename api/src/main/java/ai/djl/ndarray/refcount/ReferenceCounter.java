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
package ai.djl.ndarray.refcount;

/**
 * The ReferenceCounter interface.
 *
 * <p>This interface has been derived from {@code org.bytedeco.javacpp.Pointer} by Samuel Audet
 */
public interface ReferenceCounter {

    /** Increments the reference count by 1 starting from initially 0. */
    void retain();

    /**
     * Decrements the reference count by 1, in turn deallocating this Pointer when the count drops
     * to 0.
     *
     * @return true when the count drops to 0 and deallocation has occurred
     */
    boolean release();

    /**
     * Returns the count value.
     *
     * @return the count value
     */
    int count();
}
