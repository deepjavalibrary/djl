package com.amazon.ai.ndarray;

import com.amazon.ai.engine.Engine;

/** This class contains various methods for manipulating NDArrays. */
public final class NDArrays {

    private NDArrays() {}

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @return Returns the result of the addition
     */
    public static NDArray add(NDArray a, Number n) {
        return add(a, n, new NDFuncParams());
    }

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @param params optional params to the function
     * @return Returns the result of the addition
     */
    public static NDArray add(NDArray a, Number n, NDFuncParams params) {
        return Engine.getInstance().getEngineNDArrays().add(a, n, params);
    }

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @return Returns the result of the addition
     */
    public static NDArray add(Number n, NDArray a) {
        return add(n, a, new NDFuncParams());
    }

    /**
     * Adds a number to each element of an {@link NDArray}.
     *
     * @param a the NDArray that will be added to.
     * @param n the number to add to the {@link NDArray} elements.
     * @param params optional params to the function
     * @return Returns the result of the addition
     */
    public static NDArray add(Number n, NDArray a, NDFuncParams params) {
        return Engine.getInstance().getEngineNDArrays().add(n, a, params);
    }

    /**
     * Adds two {@link NDArray}s with broadcasting.
     *
     * @param a the left NDArray
     * @param b the right NDArray
     * @return Returns the result of the addition
     */
    public static NDArray add(NDArray a, NDArray b) {
        return add(a, b, new NDFuncParams());
    }

    /**
     * Adds two {@link NDArray}s with broadcasting.
     *
     * @param a the left NDArray
     * @param b the right NDArray
     * @param params optional params to the function
     * @return Returns the result of the addition
     */
    public static NDArray add(NDArray a, NDArray b, NDFuncParams params) {
        return Engine.getInstance().getEngineNDArrays().add(a, b, params);
    }
}
