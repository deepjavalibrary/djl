/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.IValue;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * A special {@link NDArray} that carries a prebuilt {@code Dict[str, Tensor]} {@link IValue}.
 *
 * <p>It is only a transport for model inputs. It does not support any tensor operation.
 */
public final class PtIValueNDArray extends NDArrayAdapter {

    private IValue ivalue;
    private String[] keys;
    private long[][] shapes;

    /**
     * Constructs a carrier that takes ownership of the {@link IValue}.
     *
     * @param manager the manager that owns this carrier
     * @param name the name of the carrier
     * @param ivalue the dictionary {@link IValue}
     * @param keys the dictionary keys
     * @param shapes the tensor shape for each key
     */
    public PtIValueNDArray(
            PtNDManager manager, String name, IValue ivalue, String[] keys, long[][] shapes) {
        super(manager, null, null, DataType.UNKNOWN, NDManager.nextUid());
        setName(name);
        this.ivalue = ivalue;
        this.keys = keys.clone();
        this.shapes = shapes.clone();
        manager.attachInternal(getUid(), this);
    }

    /**
     * Returns the dictionary {@link IValue}.
     *
     * @return the dictionary {@link IValue}
     * @throws IllegalStateException if this carrier is closed
     */
    public IValue getIValue() {
        if (isClosed) {
            throw new IllegalStateException("PtIValueNDArray has been closed");
        }
        return ivalue;
    }

    /**
     * Returns the dictionary keys.
     *
     * @return the dictionary keys
     */
    public String[] getKeys() {
        return keys.clone();
    }

    /**
     * Returns the tensor shapes, one per key in key order.
     *
     * @return the tensor shapes
     */
    public Shape[] getShapes() {
        Shape[] result = new Shape[shapes.length];
        for (int i = 0; i < result.length; ++i) {
            result[i] = new Shape(shapes[i]);
        }
        return result;
    }

    private static UnsupportedOperationException unsupported(String op) {
        return new UnsupportedOperationException(
                "PtIValueNDArray carries a Dict[str, Tensor] and does not support " + op);
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        throw unsupported("getShape");
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        throw unsupported("getDataType");
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer(boolean tryDirect) {
        throw unsupported("toByteBuffer");
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        throw unsupported("intern");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray duplicate() {
        throw unsupported("duplicate");
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = PtNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        ivalue.close();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "PtIValueNDArray{name=" + getName() + ", keys=" + Arrays.toString(keys) + "}";
    }
}
