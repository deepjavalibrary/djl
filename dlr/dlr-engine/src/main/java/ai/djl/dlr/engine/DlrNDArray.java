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
package ai.djl.dlr.engine;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.UUID;

/** {@code DlrNDArray} is the DLR implementation of {@link NDArray}. */
public class DlrNDArray implements NDArrayAdapter {

    private DlrNDManager manager;
    private ByteBuffer data;
    private Shape shape;
    private String name;
    private boolean isClosed;
    private String uid;

    /**
     * Constructs an DLR NDArray from a {@link DlrNDManager} (internal. Use {@link NDManager}
     * instead).
     *
     * @param manager the manager to attach the new array to
     * @param data the underlying data
     * @param shape the shape of {@code DlrNDArray}
     */
    DlrNDArray(DlrNDManager manager, ByteBuffer data, Shape shape) {
        this.manager = manager;
        this.data = data;
        this.shape = shape;
        uid = UUID.randomUUID().toString();
        manager.attachInternal(uid, this);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
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
        // DLR only supports float32
        return DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        // TODO: support GPU
        return Device.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = (DlrNDManager) manager;
        manager.attachInternal(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttach(NDManager manager) {
        detach();
        NDManager original = this.manager;
        this.manager = (DlrNDManager) manager;
        manager.tempAttachInternal(original, getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = DlrNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        data.rewind();
        return data;
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
        isClosed = true;
    }
}
