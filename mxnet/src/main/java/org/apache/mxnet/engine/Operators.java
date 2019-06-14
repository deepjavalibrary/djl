package org.apache.mxnet.engine;

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;
import com.amazon.ai.util.PairList;
import java.util.Map;
import org.apache.mxnet.jna.FunctionInfo;
import org.apache.mxnet.jna.JnaUtils;

public final class Operators {

    private static final Map<String, FunctionInfo> OPS = JnaUtils.getNdArrayFunctions();

    private Operators() {
        // not callable
    }

    /**
     * fill target NDArray with ones.
     *
     * @param factory {@link MxNDFactory} used to created the target NDArray
     * @param shape {@link Shape} of the target NDArray
     * @param context {@link Context} of the target NDArray
     * @param dataType {@link DataType} of the target NDArray
     * @param sparseFormat {@link SparseFormat} of the target NDArray
     * @return target NDArray with ones
     */
    public static MxNDArray ones(
            MxNDFactory factory,
            Shape shape,
            Context context,
            DataType dataType,
            SparseFormat sparseFormat) {
        return fill(factory, shape, context, dataType, sparseFormat, "ones");
    }

    /**
     * fill target NDArray with zeros.
     *
     * @param factory {@link MxNDFactory} used to created the target NDArray
     * @param shape {@link Shape} of the target NDArray
     * @param context {@link Context} of the target NDArray
     * @param dataType {@link DataType} of the target NDArray
     * @param sparseFormat {@link SparseFormat} of the target NDArray
     * @return target NDArray with zeros
     */
    public static MxNDArray zeros(
            MxNDFactory factory,
            Shape shape,
            Context context,
            DataType dataType,
            SparseFormat sparseFormat) {
        return fill(factory, shape, context, dataType, sparseFormat, "zeros");
    }

    private static MxNDArray fill(
            MxNDFactory factory,
            Shape shape,
            Context context,
            DataType dataType,
            SparseFormat sparseFormat,
            String opName) {
        PairList<String, String> params = new PairList<>();
        if (shape == null) {
            throw new NullPointerException(
                    String.format("Shape is required for %s operator", opName));
        }
        addParamHelper(params, shape, null);
        addParamHelper(params, context, null);
        addParamHelper(params, dataType, null);
        addParamHelper(params, sparseFormat, null);
        FunctionInfo functionInfo = OPS.get("_" + opName);
        return functionInfo.invoke(factory, params)[0];
    }

    /**
     * Return an array of zeros with the same {@link Shape}, {@link DataType} and {@link
     * SparseFormat} as the input array. The storage type of zerosLike output depends on the storage
     * type of the input
     *
     * @param factory {@link MxNDFactory} used to created the target NDArray
     * @param input input {@link MxNDArray}
     * @return output {@link MxNDArray} of zeros
     */
    public static MxNDArray zerosLike(MxNDFactory factory, MxNDArray input) {
        FunctionInfo functionInfo = OPS.get("zeros_like");
        return functionInfo.invoke(factory, input, null)[0];
    }

    /**
     * Return an array of ones with the same {@link Shape}, {@link DataType} and {@link
     * SparseFormat} as the input array. The storage type of onesLike output depends on the storage
     * type of the input
     *
     * @param factory {@link MxNDFactory} used to created the target NDArray
     * @param input input {@link MxNDArray}
     * @return output {@link MxNDArray} of zeros
     */
    public static MxNDArray onesLike(MxNDFactory factory, MxNDArray input) {
        FunctionInfo functionInfo = OPS.get("ones_like");
        return functionInfo.invoke(factory, input, null)[0];
    }

    public static MxNDArray argsort(
            MxNDFactory factory, MxNDArray input, Integer axis, Boolean isAscend) {
        PairList<String, String> params = new PairList<>();
        addParamHelper(params, axis, "axis");
        addParamHelper(params, isAscend, "is_ascend");
        FunctionInfo functionInfo = OPS.get("argsort");
        return functionInfo.invoke(factory, input, params)[0];
    }

    public static MxNDArray softmax(
            MxNDFactory factory, MxNDArray input, Integer axis, Double temperature) {
        PairList<String, String> params = new PairList<>();
        addParamHelper(params, axis, "axis");
        addParamHelper(params, temperature, "temperature");
        FunctionInfo functionInfo = OPS.get("softmax");
        return functionInfo.invoke(factory, input, params)[0];
    }

    public static MxNDArray[] split(
            MxNDFactory factory,
            MxNDArray input,
            int numOutputs,
            Integer axis,
            Boolean squeezeAxis) {
        PairList<String, String> params = new PairList<>();
        addParamHelper(params, numOutputs, "num_outputs");
        addParamHelper(params, axis, "axis");
        addParamHelper(params, squeezeAxis, "squeeze_axis");

        FunctionInfo functionInfo = OPS.get("split");
        return functionInfo.invoke(factory, input, params);
    }

    /**
     * Helper function to add Objects as parameters.
     *
     * @param params The params PairList to be updated
     * @param object The object argument from operators
     * @param paramName The name to use in params PairList
     */
    public static void addParamHelper(
            PairList<String, String> params, Object object, String paramName) {
        if (object != null) {
            if (object instanceof Shape) {
                params.add("shape", object.toString());
            } else if (object instanceof Context) {
                params.add("ctx", object.toString());
            } else if (object instanceof DataType) {
                params.add("dtype", object.toString());
            } else if (object instanceof SparseFormat) {
                params.add("stype", object.toString());
            } else if (object instanceof Integer
                    || object instanceof Long
                    || object instanceof Double
                    || object instanceof Float) {
                params.add(paramName, String.valueOf(object));
            } else if (object instanceof Boolean) {
                params.add(paramName, (boolean) object ? "True" : "False");
            } else {
                throw new IllegalArgumentException(
                        "Unknown parameter type,"
                                + "please define your own paramName and String representation"
                                + "of param value and update params using params.add(name, value)");
            }
        }
    }
}
