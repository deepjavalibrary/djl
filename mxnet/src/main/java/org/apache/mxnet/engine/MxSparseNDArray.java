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
package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.types.SparseFormat;

public class MxSparseNDArray extends MxNDArray {

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

    @Override
    public ByteBuffer toByteBuffer() {
        return toDense().toByteBuffer();
    }
}
