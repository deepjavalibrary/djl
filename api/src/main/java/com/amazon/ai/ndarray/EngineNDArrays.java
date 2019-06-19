package com.amazon.ai.ndarray;

/** An internal class used for the utilities in {@link NDArrays}. */
public abstract class EngineNDArrays {

    public abstract NDArray add(NDArray a, Number n, NDFuncParams params);

    public abstract NDArray add(Number n, NDArray a, NDFuncParams params);

    public abstract NDArray add(NDArray a, NDArray b, NDFuncParams params);
}
