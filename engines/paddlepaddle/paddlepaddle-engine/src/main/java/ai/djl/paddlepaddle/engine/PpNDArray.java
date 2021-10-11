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
package ai.djl.paddlepaddle.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.jni.JniUtils;
import com.sun.jna.Pointer;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicReference;

/** {@code PpNDArray} is the PaddlePaddle implementation of {@link NDArray}. */
public class PpNDArray extends NDArrayAdapter {

    // we keep the data to prevent GC from early collecting native memory
    private ByteBuffer data;

    private AtomicReference<Long> handle;

    /**
     * Constructs an PpNDArray from a native handle (internal. Use {@link NDManager} instead).
     *
     * @param manager the manager to attach the new array to
     * @param alternativeManager the alternative manager if available
     * @param data bytebuffer that holds the native memory
     * @param handle the pointer to the native MxNDArray memory
     */
    PpNDArray(NDManager manager, NDManager alternativeManager, ByteBuffer data, long handle) {
        super(manager, alternativeManager, null, null, String.valueOf(handle));
        this.data = data;
        this.handle = new AtomicReference<>(handle);
        manager.attachInternal(uid, this);
    }

    /**
     * Sets the Level-of-Detail field of the NDArray.
     *
     * <p>checkout https://www.bookstack.cn/read/PaddlePaddle-1.3-fluid/27.md
     *
     * @param lod the Level-of-Detail representation
     */
    public void setLoD(long[][] lod) {
        JniUtils.setNdLoD(this, lod);
    }

    /**
     * Gets the Level-of-Detail field of the NDArray.
     *
     * @return the Level-of-Detail representation
     */
    public long[][] getLoD() {
        return JniUtils.getNdLoD(this);
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return JniUtils.getNameFromNd(this);
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {
        JniUtils.setNdName(this, name);
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        if (dataType == null) {
            dataType = JniUtils.getDTypeFromNd(this);
        }
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            shape = JniUtils.getShapeFromNd(this);
        }
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteNd(pointer);
        }
        this.data = ((PpNDArray) replaced).data;
        this.handle = ((PpNDArray) replaced).handle;
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = PpNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        if (data == null) {
            data = JniUtils.getByteBufferFromNd(this);
        }
        data.rewind();
        return data;
    }

    /**
     * Gets the {@link Pointer} to this resource.
     *
     * @return the {@link Pointer} to this resource
     */
    public long getHandle() {
        Long reference = handle.get();
        if (reference == null) {
            throw new IllegalStateException("Native resource has been release already.");
        }
        return reference;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteNd(pointer);
            data = null;
        }
    }
}
