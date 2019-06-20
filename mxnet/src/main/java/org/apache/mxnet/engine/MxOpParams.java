package org.apache.mxnet.engine;

import software.amazon.ai.Context;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.ndarray.types.SparseFormat;
import software.amazon.ai.util.PairList;

/** Helper for creating the MXNet operator parameters. */
public class MxOpParams extends PairList<String, String> {

    public void setShape(Shape shape) {
        if (shape != null) {
            setParam("shape", shape.toString());
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
        add(paramName, new Shape(tuple).toString());
    }
}
