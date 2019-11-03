/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray.types;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;

/**
 * A data descriptor class that encapsulates information of a {@link NDArray}.
 *
 * <p>The information includes:
 *
 * <ul>
 *   <li>Optional name of the NDArray
 *   <li>{@link Device}
 *   <li>{@link Shape}
 *   <li>{@link DataType}
 *   <li>{@link SparseFormat}
 * </ul>
 */
public class DataDesc {

    private String name;
    private Shape shape;
    private DataType dataType;

    /**
     * Constructs and initializes a {@code DataDesc} with specified {@link Shape}.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     */
    public DataDesc(Shape shape) {
        this(shape, DataType.FLOAT32, null);
    }

    /**
     * Constructs and initializes a {@code DataDesc} with specified {@link Shape} and name.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param name the name of the {@link NDArray}
     */
    public DataDesc(Shape shape, String name) {
        this(shape, DataType.FLOAT32, name);
    }

    /**
     * Constructs and initializes a {@code DataDesc} with specified {@link Shape} and {@link
     * DataType}.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     */
    public DataDesc(Shape shape, DataType dataType) {
        this(shape, dataType, null);
    }

    /**
     * Constructs and initializes a {@code DataDesc} with specified {@link Shape}, {@link DataType},
     * name, {@link Device} and {@link SparseFormat}.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param name the name of the {@link NDArray}
     */
    public DataDesc(Shape shape, DataType dataType, String name) {
        this.name = name;
        this.shape = shape;
        this.dataType = dataType;
    }

    /**
     * Returns the name of the {@link NDArray}.
     *
     * @return the name of the {@link NDArray}
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the name of the {@link NDArray}.
     *
     * @param name the name of the {@link NDArray}
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the {@link Shape} of the {@link NDArray}.
     *
     * @return the {@link Shape} of the {@link NDArray}
     */
    public Shape getShape() {
        return shape;
    }

    /**
     * Sets the {@link Shape} of the {@link NDArray}.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     */
    public void setShape(Shape shape) {
        this.shape = shape;
    }

    /**
     * Returns the {@link DataType} of the {@link NDArray}.
     *
     * @return the {@link DataType} of the {@link NDArray}
     */
    public DataType getDataType() {
        return dataType;
    }

    /**
     * Sets the {@link DataType} of the {@link NDArray}.
     *
     * @param dataType the {@link DataType} of the {@link NDArray}
     */
    public void setDataType(DataType dataType) {
        this.dataType = dataType;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return name + " shape: " + shape + " dataType: " + dataType;
    }
}
