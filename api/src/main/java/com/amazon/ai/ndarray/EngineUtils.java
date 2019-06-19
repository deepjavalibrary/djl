package com.amazon.ai.ndarray;

/** An internal class used for engine specific utilities found in classes like {@link NDArrays}. */
public abstract class EngineUtils {

    public abstract NDArray add(NDArray a, Number n, NDFuncParams fparams);

    public abstract NDArray add(Number n, NDArray a, NDFuncParams fparams);

    public abstract NDArray add(NDArray a, NDArray b, NDFuncParams fparams);
}
