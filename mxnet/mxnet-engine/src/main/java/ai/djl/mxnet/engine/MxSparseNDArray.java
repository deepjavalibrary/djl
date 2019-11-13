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
package ai.djl.mxnet.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.SparseFormat;
import com.sun.jna.Pointer;
import java.nio.Buffer;
import java.nio.ByteBuffer;

/**
 * {@code MxSparseNDArray} is an instance of {@link MxNDArray} and {@link NDArray} for sparse
 * NDArrays.
 *
 * <p>{@code MxSparseNDArray}s are created automatically when the Engine creates Arrays that are
 * sparse. They can be created deliberately by specifying the {@link SparseFormat}. Some operations
 * may not be supported with Sparse NDArrays in MXNet.
 *
 * @see SparseFormat
 */
public class MxSparseNDArray extends MxNDArray {

    /**
     * Constructs a {@code MxSparseNDArray} for the given data.
     *
     * @param manager the manager to attach the array to
     * @param handle the pointer to the native memory of the MXNDArray
     * @param fmt the sparse format
     */
    MxSparseNDArray(MxNDManager manager, Pointer handle, SparseFormat fmt) {
        super(manager, handle, fmt);
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new IllegalStateException("Unsupported operation for Sparse");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        return toDense().get(index);
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return toDense().toByteBuffer();
    }
}
