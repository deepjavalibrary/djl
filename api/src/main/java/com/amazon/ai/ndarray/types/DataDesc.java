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

public class DataDesc {

    private Context context;
    private String name;
    private Shape shape;
    private DataType dataType;
    private SparseFormat sparseFormat;
    private Layout layout;
    private int index;

    public DataDesc(Shape shape) {
        this(shape, DataType.FLOAT32, null, Layout.UNDEFINED, null, SparseFormat.DEFAULT);
    }

    public DataDesc(Shape shape, String name) {
        this(shape, DataType.FLOAT32, name, Layout.UNDEFINED, null, SparseFormat.DEFAULT);
    }

    public DataDesc(Shape shape, DataType dataType) {
        this(shape, dataType, null, Layout.UNDEFINED, null, SparseFormat.DEFAULT);
    }

    public DataDesc(Shape shape, DataType dataType, String name) {
        this(shape, dataType, name, Layout.UNDEFINED, null, SparseFormat.DEFAULT);
    }

    public DataDesc(Shape shape, DataType dataType, String name, Layout layout) {
        this(shape, dataType, name, layout, null, SparseFormat.DEFAULT);
    }

    public DataDesc(
            Shape shape,
            DataType dataType,
            String name,
            Layout layout,
            Context context,
            SparseFormat sparseFormat) {
        this.context = context;
        this.name = name;
        this.shape = shape;
        this.dataType = dataType;
        this.sparseFormat = sparseFormat;
        this.layout = layout;
    }

    public Context getContext() {
        return context;
    }

    public void setContext(Context context) {
        this.context = context;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Shape getShape() {
        return shape;
    }

    public void setShape(Shape shape) {
        this.shape = shape;
    }

    public DataType getDataType() {
        return dataType;
    }

    public void setDataType(DataType dataType) {
        this.dataType = dataType;
    }

    public SparseFormat getSparseFormat() {
        return sparseFormat;
    }

    public void setSparseFormat(SparseFormat sparseFormat) {
        this.sparseFormat = sparseFormat;
    }

    public Layout getLayout() {
        return layout;
    }

    public void setLayout(Layout layout) {
        this.layout = layout;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
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
