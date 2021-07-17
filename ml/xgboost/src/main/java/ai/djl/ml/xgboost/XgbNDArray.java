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

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;
import ml.dmlc.xgboost4j.java.JniUtils;

/** {@code XgbNDArray} is the XGBoost implementation of {@link NDArray}. */
public class XgbNDArray implements NDArrayAdapter {

    private AtomicLong handle;
    private String name;
    private String uid;
    private ByteBuffer data;
    private XgbNDManager manager;
    private Shape shape;
    private boolean isClosed;
    private SparseFormat format;

    XgbNDArray(XgbNDManager manager, long handle, Shape shape, SparseFormat format) {
        this.handle = new AtomicLong(handle);
        this.uid = String.valueOf(handle);
        this.manager = manager;
        this.manager.attachInternal(uid, this);
        this.shape = shape;
        this.format = format;
    }

    XgbNDArray(XgbNDManager manager, ByteBuffer data, Shape shape) {
        this.manager = manager;
        this.uid = UUID.randomUUID().toString();
        this.manager.attachInternal(uid, this);
        this.shape = shape;
        this.data = data;
        this.format = SparseFormat.DENSE;
    }

    /** {@inheritDoc} */
    public long getHandle() {
        return handle.get();
    }

    /** {@inheritDoc} */
    @Override
    public XgbNDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return name;
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /** {@inheritDoc} */
    @Override
    public String getUid() {
        return uid;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return Device.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        return shape;
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
        return data;
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = (XgbNDManager) manager;
        manager.attachInternal(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttach(NDManager manager) {
        detach();
        NDManager original = this.manager;
        this.manager = (XgbNDManager) manager;
        manager.tempAttachInternal(original, getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = XgbNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (isClosed) {
            return "This array is already closed";
        }
        return "ND: "
                + getShape()
                + ' '
                + getDevice()
                + ' '
                + getDataType()
                + '\n'
                + Arrays.toString(toArray());
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (handle != null && handle.get() != 0L) {
            long pointer = handle.getAndSet(0L);
            JniUtils.deleteDMatrix(pointer);
        }
        if (data != null) {
            data = null;
        }
        manager.detachInternal(getUid());
        isClosed = true;
    }
}
