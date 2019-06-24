package org.apache.mxnet.engine;

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;
import com.amazon.ai.util.PairList;

/** Helper for creating the MXNet operator parameters. */
public class MxOpParams extends PairList<String, String> {

    public void addShape(Shape shape) {
        if (shape != null) {
            add("shape", shape.toString());
        }
    }

    public void addContext(Context context) {
        if (context != null) {
            add("ctx", context.toString());
        }
    }

    public void addDataType(DataType dataType) {
        if (dataType != null) {
            add("dtype", dataType.toString());
        }
    }

    public void addSparseFormat(SparseFormat sparseFormat) {
        if (sparseFormat != null) {
            addParam("stype", sparseFormat.getValue());
        }
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
}
