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

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.jni.JniUtils;
import ai.djl.util.NativeResource;
import java.nio.ByteBuffer;

/** {@code PpNDArray} is the PaddlePaddle implementation of {@link NDArray}. */
public class PpNDArray extends NativeResource<Long> implements NDArrayAdapter {

    private PpNDManager manager;
    // we keep the data to prevent GC from early collecting native memory
    private ByteBuffer data;
    private Shape shape;
    private DataType dataType;

    /**
     * Constructs an PpNDArray from a native handle (internal. Use {@link NDManager} instead).
     *
     * @param manager the manager to attach the new array to
     * @param data bytebuffer that holds the native memory
     * @param handle the pointer to the native MxNDArray memory
     */
    public PpNDArray(PpNDManager manager, ByteBuffer data, long handle) {
        super(handle);
        this.manager = manager;
        this.data = data;
        manager.attachInternal(getUid(), this);
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
    public NDManager getManager() {
        return manager;
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
    public Device getDevice() {
        return Device.cpu();
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
    public void attach(NDManager manager) {
        detach();
        this.manager = (PpNDManager) manager;
        manager.attachInternal(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttach(NDManager manager) {
        detach();
        NDManager original = this.manager;
        this.manager = (PpNDManager) manager;
        manager.tempAttachInternal(original, getUid(), this);
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

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (isReleased()) {
            return "This array is already closed";
        }
        return toDebugString();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteNd(pointer);
            data = null;
        }
    }
}
