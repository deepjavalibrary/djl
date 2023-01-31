/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.util.passthrough;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.nio.ByteBuffer;

/**
 * An {@link NDArray} that stores an arbitrary Java object.
 *
 * <p>This class is mainly for use in extensions and hybrid engines. Despite it's name, it will
 * often not contain actual {@link NDArray}s but just any object necessary to conform to the DJL
 * predictor API.
 */
public class PassthroughNDArray extends NDArrayAdapter {

    private Object object;

    /**
     * Constructs a {@link PassthroughNDArray} storing an object.
     *
     * @param object the object to store
     */
    public PassthroughNDArray(Object object) {
        super(PassthroughNDManager.INSTANCE, null, null, null, null);
        this.object = object;
    }

    /**
     * Constructs a {@link PassthroughNDArray} storing an object.
     *
     * @param object the object to store
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     */
    public PassthroughNDArray(Object object, Shape shape, DataType dataType) {
        super(PassthroughNDManager.INSTANCE, null, shape, dataType, null);
        this.object = object;
    }

    /**
     * Returns the object stored.
     *
     * @return the object stored
     */
    public Object getObject() {
        return object;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        if (object instanceof ByteBuffer) {
            return (ByteBuffer) object;
        }
        throw new UnsupportedOperationException("Operation not supported for FastText");
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        throw new UnsupportedOperationException("Operation not supported for FastText");
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {}
}
