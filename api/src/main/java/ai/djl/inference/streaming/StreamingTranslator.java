/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.inference.streaming;

import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.stream.Stream;

/**
 * An expansion of the {@link Translator} with postProcessing for the {@link StreamingBlock} (used
 * by {@link ai.djl.inference.Predictor#streamingPredict(Object)}.
 *
 * @param <I> the input type
 * @param <O> the output type
 */
public interface StreamingTranslator<I, O> extends Translator<I, O> {

    /**
     * Processes the output NDList to the corresponding output object.
     *
     * @param ctx the toolkit used for post-processing
     * @param list the output NDList after inference, usually immutable in engines like
     *     PyTorch. @see <a href="https://github.com/deepjavalibrary/djl/issues/1774">Issue 1774</a>
     * @return the output object of expected type
     * @throws Exception if an error occurs during processing output
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    StreamOutput<O> processStreamOutput(TranslatorContext ctx, Stream<NDList> list)
            throws Exception;

    /**
     * Returns what kind of {@link StreamOutput} this {@link StreamingTranslator} supports.
     *
     * @return what kind of {@link StreamOutput} this {@link StreamingTranslator} supports
     */
    Support getSupport();

    /**
     * Returns whether the {@link StreamingTranslator} supports iterative output.
     *
     * @return whether the {@link StreamingTranslator} supports iterative output
     * @see StreamOutput#getIterativeOutput()
     */
    default boolean supportsIterative() {
        return getSupport().iterative();
    }

    /**
     * Returns whether the {@link StreamingTranslator} supports iterative output.
     *
     * @return whether the {@link StreamingTranslator} supports iterative output
     * @see StreamOutput#getAsyncOutput()
     */
    default boolean supportsAsync() {
        return getSupport().async();
    }

    /**
     * A {@link StreamOutput} represents a streamable output type (either iterative or
     * asynchronous).
     *
     * <p>There are two modes for the {@link StreamOutput}. When using it, you must choose one of
     * the modes and can only access it once. Any other usage including trying both modes or trying
     * one mode twice will result in an {@link IllegalStateException}.
     *
     * <p>The first mode is the iterative mode which can be used by calling {@link
     * #getIterativeOutput()}, it returns a result that has an internal iterate method. When calling
     * the iterating method, it will compute an additional part of the output.
     *
     * <p>The second mode is asynchronous mode. Here, you can produce a mutable output object by
     * calling {@link #getAsyncOutput()}. Then, calling {@link #computeAsyncOutput()} will
     * synchronously compute the results and deposit them into the prepared output. This method
     * works best with manual threading where the worker can return the template result to another
     * thread and then continue to compute it.
     *
     * @param <O> the output type
     */
    abstract class StreamOutput<O> {
        private O output;
        private boolean computed;

        /**
         * Returns a template object to be used with the async output.
         *
         * <p>This should only be an empty data structure until {@link #computeAsyncOutput()} is
         * called.
         *
         * @return a template object to be used with the async output
         */
        public final O getAsyncOutput() {
            if (output != null) {
                throw new IllegalStateException("The StreamOutput can only be gotten once");
            }
            if (computed) {
                throw new IllegalStateException(
                        "Attempted to getAsyncOutput, but has already called getIterativeOutput."
                                + " Only one kind of output can be used.");
            }
            output = buildAsyncOutput();
            return output;
        }

        /**
         * Performs the internal implementation of {@link #getAsyncOutput()}.
         *
         * @return the output for {@link #getAsyncOutput()}.
         */
        protected abstract O buildAsyncOutput();

        /**
         * Computes the actual value and stores it in the object returned earlier by {@link
         * #getAsyncOutput()}.
         */
        public final void computeAsyncOutput() {
            if (output == null) {
                throw new IllegalStateException(
                        "Before calling computeAsyncOutput, you must first getAsyncOutput");
            }
            if (computed) {
                throw new IllegalStateException("Attempted to computeAsyncOutput multiple times.");
            }
            computed = true;
            computeAsyncOutputInternal(output);
        }

        /**
         * Performs the internal implementation of {@link #computeAsyncOutput()}.
         *
         * @param output the output object returned by the earlier call to {@link
         *     #getAsyncOutput()}.
         */
        protected abstract void computeAsyncOutputInternal(O output);

        /**
         * Returns an iterative streamable output.
         *
         * @return an iterative streamable output
         */
        public final O getIterativeOutput() {
            if (output != null) {
                throw new IllegalStateException(
                        "Can't call getIterativeOutput after already using getAsyncOutput.");
            }
            if (computed) {
                throw new IllegalStateException(
                        "Attempted to getIterativeOutput multiple times. getIterativeOutput can"
                                + " only be called once");
            }
            return getIterativeOutputInternal();
        }

        /**
         * Performs the internal implementation of {@link #getIterativeOutput()}.
         *
         * @return the output for {@link #getIterativeOutput()}
         */
        public abstract O getIterativeOutputInternal();
    }

    /** What types of {@link StreamOutput}s are supported by a {@link StreamingTranslator}. */
    enum Support {
        /** Supports {@link #iterative()} but not {@link #async()}. */
        ITERATIVE(true, false),

        /** Supports {@link #async()} but not {@link #iterative()}. */
        ASYNC(false, true),

        /** Supports both {@link #iterative()} and {@link #async()}. */
        BOTH(true, true);

        private boolean iterative;
        private boolean async;

        Support(boolean iterative, boolean async) {
            this.iterative = iterative;
            this.async = async;
        }

        /**
         * Returns whether the {@link StreamingTranslator} supports iterative output.
         *
         * @return whether the {@link StreamingTranslator} supports iterative output
         * @see StreamOutput#getIterativeOutput()
         */
        public boolean iterative() {
            return iterative;
        }

        /**
         * Returns whether the {@link StreamingTranslator} supports iterative output.
         *
         * @return whether the {@link StreamingTranslator} supports iterative output
         * @see StreamOutput#getAsyncOutput()
         */
        public boolean async() {
            return async;
        }
    }
}
