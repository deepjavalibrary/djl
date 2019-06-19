package org.apache.mxnet.engine;

import com.amazon.ai.ndarray.EngineNDArrays;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFuncParams;

public class MxEngineNDArrays extends EngineNDArrays {

    @Override
    public NDArray add(NDArray a, Number n, NDFuncParams params) {
        return null;
    }

    @Override
    public NDArray add(Number n, NDArray a, NDFuncParams params) {
        return null;
    }

    @Override
    public NDArray add(NDArray a, NDArray b, NDFuncParams params) {
        return null;
    }
}
