/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import software.amazon.ai.Context;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.ndarray.types.SparseFormat;
import software.amazon.ai.util.PairList;

/** Helper for creating the MXNet operator parameters. */
public class MxOpParams extends PairList<String, String> {

    public void setShape(Shape shape) {
        setShape("shape", shape);
    }

    public void setShape(String alias, Shape shape) {
        if (shape != null) {
            setParam(alias, shape.toString());
        }
    }

    public void setContext(Context context) {
        if (context != null) {
            setParam("ctx", context.toString());
        }
    }

    public void setDataType(DataType dataType) {
        if (dataType != null) {
            setParam("dtype", dataType.getType());
        }
    }

    public void setSparseFormat(SparseFormat sparseFormat) {
        if (sparseFormat != null) {
            setParam("stype", String.valueOf(sparseFormat.getValue()));
        }
    }

    public void setParam(String paramName, String value) {
        remove(paramName);
        add(paramName, value);
    }

    public void addParam(String paramName, int value) {
        add(paramName, String.valueOf(value));
    }

    public void addParam(String paramName, long value) {
        add(paramName, String.valueOf(value));
    }

    public void addParam(String paramName, double value) {
        add(paramName, String.valueOf(value));
    }

    public void addParam(String paramName, float value) {
        add(paramName, String.valueOf(value));
    }

    public void addParam(String paramName, boolean value) {
        add(paramName, value ? "True" : "False");
    }

    public void addTupleParam(String paramName, int... tuple) {
        StringBuilder sb = new StringBuilder();
        sb.append('(');
        for (int i = 0; i < tuple.length; ++i) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(tuple[i]);
        }
        sb.append(')');
        add(paramName, sb.toString());
    }

    public void addTupleParam(String paramName, long... tuple) {
        StringBuilder sb = new StringBuilder();
        sb.append('(');
        for (int i = 0; i < tuple.length; ++i) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(tuple[i]);
        }
        sb.append(')');
        add(paramName, sb.toString());
    }
}
