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
package ai.djl.pytorch.engine;

import ai.djl.ndarray.refcount.Deallocator;
import ai.djl.ndarray.refcount.DeallocatorReference;

/**
 * A {@link Deallocator} that calls, during garbage collection, the method {@link
 * PtNDArray#deallocate()} from the referenceCountedObject of type {@link PtNDArray}.
 *
 * <p>This class has been derived from {@code org.bytedeco.javacpp.Pointer} by Samuel Audet
 */
public class PtNDArrayDeallocator extends DeallocatorReference {
    PtNDArray referenceCountedObject;

    /**
     * Constructs and initializes a {@code PtNDArrayDeallocator} with a {@link PtNDArray} to.
     *
     * @param p - the {@link PtNDArray} to be deallocated
     */
    public PtNDArrayDeallocator(PtNDArray p) {
        super(p, null);
        this.deallocator = this;
        this.referenceCountedObject = p;
    }

    /** {@inheritDoc} */
    @Override
    public void deallocate() {
        PtNDArray.deallocate(referenceCountedObject);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return getClass().getName() + "[referenceCountedObject=" + referenceCountedObject + "]";
    }
}
