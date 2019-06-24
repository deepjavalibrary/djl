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
package com.amazon.ai.ndarray.types;

import com.amazon.ai.Context;

/**
 * A data descriptor class encapsulate information of a {@link com.amazon.ai.ndarray.NDArray}.
 *
 * <p>The information includes:
 *
 * <ul>
 *   <li>Optional name of the NDArray
 *   <li>{@link com.amazon.ai.Context}
 *   <li>{@link com.amazon.ai.ndarray.types.Shape}
 *   <li>{@link com.amazon.ai.ndarray.types.DataType}
 *   <li>{@link com.amazon.ai.ndarray.types.SparseFormat}
 *   <li>{@link com.amazon.ai.ndarray.types.Layout}
 * </ul>
 */
public class DataDesc {

    private Context context;
    private String name;
    private Shape shape;
    private DataType dataType;
    private Layout layout;

    /**
     * Constructs and initializes a <code>DataDesc</code> with specified {@link Shape}.
     *
     * @param shape the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public DataDesc(Shape shape) {
        this(shape, DataType.FLOAT32, null, Layout.UNDEFINED, null);
    }

    /**
     * Constructs and initializes a <code>DataDesc</code> with specified {@link Shape} and name.
     *
     * @param shape the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param name the name of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public DataDesc(Shape shape, String name) {
        this(shape, DataType.FLOAT32, name, Layout.UNDEFINED, null);
    }

    /**
     * Constructs and initializes a <code>DataDesc</code> with specified {@link Shape} and {@link
     * DataType}.
     *
     * @param shape the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param dataType the {@link DataType} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public DataDesc(Shape shape, DataType dataType) {
        this(shape, dataType, null, Layout.UNDEFINED, null);
    }

    /**
     * Constructs and initializes a <code>DataDesc</code> with specified {@link Shape}, {@link
     * DataType} and name.
     *
     * @param shape the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param dataType the {@link DataType} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param name the name of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public DataDesc(Shape shape, DataType dataType, String name) {
        this(shape, dataType, name, Layout.UNDEFINED, null);
    }

    /**
     * Constructs and initializes a <code>DataDesc</code> with specified {@link Shape}, {@link
     * DataType}, name and {@link Layout}.
     *
     * @param shape the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param dataType the {@link DataType} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param name the name of the {@link com.amazon.ai.ndarray.NDArray}
     * @param layout the {@link Layout} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public DataDesc(Shape shape, DataType dataType, String name, Layout layout) {
        this(shape, dataType, name, layout, null);
    }

    /**
     * Constructs and initializes a <code>DataDesc</code> with specified {@link Shape}, {@link
     * DataType}, name, {@link Layout}, {@link Context} and {@link SparseFormat}.
     *
     * @param shape the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param dataType the {@link DataType} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param name the name of the {@link com.amazon.ai.ndarray.NDArray}
     * @param layout the {@link Layout} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param context the {@link Context} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public DataDesc(Shape shape, DataType dataType, String name, Layout layout, Context context) {
        this.context = context;
        this.name = name;
        this.shape = shape;
        this.dataType = dataType;
        this.layout = layout;
    }

    /**
     * Returns the {@link Context} of the {@link com.amazon.ai.ndarray.NDArray}.
     *
     * @return the {@link Context} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public Context getContext() {
        return context;
    }

    /**
     * Sets the {@link Context} of the {@link com.amazon.ai.ndarray.NDArray}.
     *
     * @param context the {@link Context} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public void setContext(Context context) {
        this.context = context;
    }

    /**
     * Returns the name the name of the {@link com.amazon.ai.ndarray.NDArray}.
     *
     * @return name the name of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the name the name of the {@link com.amazon.ai.ndarray.NDArray}.
     *
     * @param name the name the name of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}.
     *
     * @return the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public Shape getShape() {
        return shape;
    }

    /**
     * Sets the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}.
     *
     * @param shape the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public void setShape(Shape shape) {
        this.shape = shape;
    }

    /**
     * Returns the {@link DataType} of the {@link com.amazon.ai.ndarray.NDArray}.
     *
     * @return the {@link DataType} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public DataType getDataType() {
        return dataType;
    }

    /**
     * Sets the {@link DataType} of the {@link com.amazon.ai.ndarray.NDArray}.
     *
     * @param dataType the {@link DataType} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public void setDataType(DataType dataType) {
        this.dataType = dataType;
    }

    /**
     * Returns the {@link Layout} of the {@link com.amazon.ai.ndarray.NDArray}.
     *
     * @return the {@link Layout} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public Layout getLayout() {
        return layout;
    }

    /**
     * Sets the {@link Layout} of the {@link com.amazon.ai.ndarray.NDArray}.
     *
     * @param layout the {@link Layout} of the {@link com.amazon.ai.ndarray.NDArray}
     */
    public void setLayout(Layout layout) {
        this.layout = layout;
    }

    public int getMajorAxis() {
        return getBatchAxis(layout);
    }

    public static int getBatchAxis(Layout layout) {
        if (layout == null || Layout.UNDEFINED == layout) {
            return 0;
        }

        if (!layout.getValue().contains("N")) {
            throw new IllegalArgumentException("no Batch Axis('N') found in Layout!");
        }

        return layout.getValue().indexOf('N');
    }
}
