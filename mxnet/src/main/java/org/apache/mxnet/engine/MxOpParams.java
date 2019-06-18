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
            add("stype", sparseFormat.toString());
        }
    }

    public void addParam(String paramName, Integer i) {
        if (i != null) {
            add(paramName, String.valueOf(i));
        }
    }

    public void addParam(String paramName, Long l) {
        if (l != null) {
            add(paramName, String.valueOf(l));
        }
    }

    public void addParam(String paramName, Double d) {
        if (d != null) {
            add(paramName, String.valueOf(d));
        }
    }

    public void addParam(String paramName, Float f) {
        if (f != null) {
            add(paramName, String.valueOf(f));
        }
    }

    public void addParam(String paramName, Boolean b) {
        if (b != null) {
            add(paramName, b ? "True" : "False");
        }
    }
}
