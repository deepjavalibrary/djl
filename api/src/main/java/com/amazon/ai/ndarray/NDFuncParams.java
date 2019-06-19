package com.amazon.ai.ndarray;

import com.amazon.ai.Context;

/** A parameter class representing standard options for an NDArray function. */
public final class NDFuncParams {

    private NDFactory factory;
    private Context context;
    private boolean isInPlace;
    private NDArray out;

    public static final NDFuncParams NONE = new Builder().build();

    /**
     * Returns the factory to use for the function result otherwise null.
     *
     * @return Returns the factory to use for the function result otherwise null
     */
    public NDFactory getFactory() {
        return factory;
    }

    /**
     * Returns the specified context to use otherwise null.
     *
     * @return Returns the specified context to use otherwise null
     */
    public Context getContext() {
        return context;
    }

    /**
     * Returns whether to execute the function in place.
     *
     * @return Returns whether to execute the function in place
     */
    public boolean getInPlace() {
        return isInPlace;
    }

    /**
     * Returns whether to have the function put the result in a pre-existing {@link NDArray}.
     *
     * @return Returns whether to have the function put the result in a pre-existing {@link NDArray}
     */
    public NDArray getOut() {
        return out;
    }

    private NDFuncParams(NDFactory factory, Context context, boolean isInPlace, NDArray out) {
        this.factory = factory;
        this.context = context;
        this.isInPlace = isInPlace;
        this.out = out;
    }

    /**
     * Helper to execute a function in place.
     *
     * @return The NDFuncParam to pass to the function
     */
    public static NDFuncParams inPlace() {
        return new NDFuncParams.Builder().setInPlace(true).build();
    }

    /**
     * Helper to execute a function and put the result in a pre-existing {@link NDArray}.
     *
     * @param out the NDArray where the result should be placed (will be returned by the function)
     * @return The NDFuncParam to pass to the function
     */
    public static NDFuncParams output(NDArray out) {
        return new NDFuncParams.Builder().setOut(out).build();
    }

    /** Builder to create a {@link NDFuncParams} */
    public static class Builder {

        private NDFactory factory;
        private Context context;
        private boolean isInPlace;
        private NDArray out;

        /**
         * Sets the factory to use for the function result.
         *
         * @param factory the factory to use for the function result
         * @return Returns the Builder
         */
        public Builder setFactory(NDFactory factory) {
            this.factory = factory;
            return this;
        }

        /**
         * Sets the context to use for the function result.
         *
         * @param context the context to use for the function result
         * @return Returns the Builder
         */
        public Builder setContext(Context context) {
            this.context = context;
            return this;
        }

        /**
         * Sets whether to execute the function in place.
         *
         * @param isInPlace whether to execute the function in place
         * @return Returns the Builder
         */
        public Builder setInPlace(boolean isInPlace) {
            this.isInPlace = isInPlace;
            return this;
        }

        /**
         * Sets whether to have the function put the result in a pre-existing {@link NDArray}.
         *
         * @param out the output NDArray
         * @return Returns the Builder
         */
        public Builder setOut(NDArray out) {
            this.out = out;
            return this;
        }

        /**
         * Builds the Builder into an {@link NDFuncParams}.
         *
         * @return Returns the built {@link NDFuncParams}
         */
        public NDFuncParams build() {
            if (isInPlace && out != null) {
                throw new IllegalStateException("Can not specify output and execute in place");
            }
            return new NDFuncParams(factory, context, isInPlace, out);
        }
    }
}
