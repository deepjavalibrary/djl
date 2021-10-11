/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ml.xgboost;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import java.nio.ByteBuffer;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;
import ml.dmlc.xgboost4j.java.JniUtils;

/** {@code XgbNDArray} is the XGBoost implementation of {@link NDArray}. */
public class XgbNDArray extends NDArrayAdapter {

    private AtomicLong handle;
    private ByteBuffer data;
    private SparseFormat format;

    XgbNDArray(
            NDManager manager,
            NDManager alternativeManager,
            long handle,
            Shape shape,
            SparseFormat format) {
        super(manager, alternativeManager, shape, DataType.FLOAT32, String.valueOf(handle));
        this.handle = new AtomicLong(handle);
        this.format = format;
        manager.attachInternal(uid, this);
    }

    XgbNDArray(
            NDManager manager,
            NDManager alternativeManager,
            ByteBuffer data,
            Shape shape,
            DataType dataType) {
        super(manager, alternativeManager, shape, dataType, UUID.randomUUID().toString());
        this.data = data;
        this.format = SparseFormat.DENSE;
        manager.attachInternal(uid, this);
    }

    /** {@inheritDoc} */
    public long getHandle() {
        if (handle == null) {
            throw new UnsupportedOperationException(
                    "XgbNDArray only support float32 and shape must be in two dimension.");
        }
        return handle.get();
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        return format;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        if (data == null) {
            throw new UnsupportedOperationException("Cannot obtain value from DMatrix");
        }
        data.rewind();
        return data;
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        if (handle != null && handle.get() != 0L) {
            long pointer = handle.getAndSet(0L);
            JniUtils.deleteDMatrix(pointer);
        }
        XgbNDArray array = (XgbNDArray) replaced;
        data = array.data;
        handle = array.handle;
        format = array.format;
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = XgbNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        if (handle != null && handle.get() != 0L) {
            long pointer = handle.getAndSet(0L);
            JniUtils.deleteDMatrix(pointer);
        }
    }
}
