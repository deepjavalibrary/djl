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
package org.apache.mxnet.engine;

import org.apache.mxnet.types.DataType;
import org.apache.mxnet.types.Layout;

public class DataDesc {

    private String name;
    private Shape shape;
    private DataType dataType;
    private String layout;

    public DataDesc(String name, Shape shape, DataType dataType, String layout) {
        this.name = name;
        this.shape = shape;
        this.dataType = dataType;
        this.layout = layout;
    }

    public String getName() {
        return name;
    }

    public Shape getShape() {
        return shape;
    }

    public int getShape(char c) {
        return shape.get(layout.indexOf(c));
    }

    public DataType getDataType() {
        return dataType;
    }

    public String getLayout() {
        return layout;
    }

    public int getMajorAxis() {
        return getBatchAxis(layout);
    }

    public static int getBatchAxis(String layout) {
        if (layout == null || Layout.UNDEFINED.getValue().equals(layout)) {
            return 0;
        }

        if (!layout.contains("N")) {
            throw new IllegalArgumentException("no Batch Axis('N') found in Layout!");
        }

        return layout.indexOf('N');
    }
}
