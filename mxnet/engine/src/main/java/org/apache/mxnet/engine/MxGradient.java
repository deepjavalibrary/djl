/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package org.apache.mxnet.engine;

import java.util.stream.Collectors;
import org.apache.mxnet.jna.JnaUtils;
import software.amazon.ai.Block;
import software.amazon.ai.Parameter;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.training.Gradient;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

public interface MxGradient extends Gradient {

    /**
     * Get status on whether Autograd is recording computations.
     *
     * @return Current state of recording.
     */
    public static boolean isRecording() {
        return JnaUtils.autogradIsRecording();
    }

    /**
     * Get status on whether Autograd is in training/predicting mode.
     *
     * @return Current state of training/predicting.
     */
    public static boolean isTraining() {
        return JnaUtils.autogradIsTraining();
    }

    /**
     * Set status to recording/not recording. When recording, graph will be constructed for gradient
     * computation.
     *
     * @param isRecording recording state to be set
     * @return previous recording state before this set
     */
    public static boolean setRecording(boolean isRecording) {
        return JnaUtils.autogradSetIsRecording(isRecording);
    }

    /**
     * Set status to training/predicting. This affects ctx.is_train in operator running context. For
     * example, Dropout will drop inputs randomly when isTraining=True while simply passing through
     * if isTraining=False.
     *
     * @param isTraining {@code true} if for training
     * @return the previous status before this set
     */
    public static boolean setTraining(boolean isTraining) {
        return JnaUtils.autogradSetTraining(isTraining);
    }

    public static class Collector implements Gradient.Collector {

        public Collector() {
            boolean prevRecordingState = setRecording(true);
            if (prevRecordingState) {
                throw new IllegalStateException(
                        "Autograd Recording is already set to True. "
                                + "Please create autograd using try with resource ");
            }
            boolean prevTrainingState = setTraining(true);
            if (prevTrainingState) {
                throw new IllegalStateException(
                        "Autograd Training is already set to True. "
                                + "Please create autograd using try with resource ");
            }
        }

        /** {@inheritDoc} */
        @Override
        public void close() {
            setRecording(false);
            setTraining(false);
        }

        /**
         * Returns the gradient buffer attached to the target {@code NDArray}.
         *
         * @param array the target {@code NDArray} to get gradient buffer
         * @return the gradient buffer attached to the input {@code NDArray}
         */
        public MxNDArray getGradient(MxNDArray array) {
            return (MxNDArray) array.getGradient();
        }

        /**
         * Run backward and calculate gradient w.r.t previously marked variable (head).
         *
         * @param array target NDArray to run backward and calculate gradient w.r.t head
         */
        public void backward(MxNDArray array) {
            array.backward();
        }

        @Override
        public OptimizerKey collectFor(Optimizer optimizer) {
            for (Parameter param : optimizer.getParameters().values()) {
                param.startGradientCollection(this);
            }
            return new OptimizerKey(optimizer);
        }

        @Override
        public BlockKey collectFor(Block block) {
            for (Parameter param : block.getParameters().values()) {
                param.startGradientCollection(this);
            }
            return new BlockKey(block);
        }

        @Override
        public ParameterKey collectFor(Parameter parameter) {
            parameter.startGradientCollection(this);
            return new ParameterKey(parameter);
        }

        @Override
        public NDArrayKey collectFor(NDArray array) {
            ((MxNDArray) array).attachGradient();
            return new NDArrayKey(array);
        }

        @Override
        public void collectProgress(NDArray target) {
            ((MxNDArray) target).backward(true, true);
        }

        @Override
        public Dict collect(NDArray target) {
            ((MxNDArray) target).backward(false, true);
            setRecording(false);
            return new Dict();
        }
    }

    public static class Dict implements Gradient.Dict {

        @Override
        public Gradient.OptimizerGrad get(Gradient.OptimizerKey key) {
            return new OptimizerGrad(key.getOptimizer());
        }

        @Override
        public Gradient.BlockGrad get(Gradient.BlockKey key) {
            return new BlockGrad(key.getBlock());
        }

        @Override
        public Gradient.ParameterGrad get(Gradient.ParameterKey key) {
            return new ParameterGrad(key.getParameter());
        }

        @Override
        public Gradient.NDArrayGrad get(Gradient.NDArrayKey key) {
            return new NDArrayGrad(key.getArray());
        }
    }

    public static class OptimizerKey implements Gradient.OptimizerKey {

        private Optimizer optimizer;

        public OptimizerKey(Optimizer optimizer) {
            this.optimizer = optimizer;
        }

        @Override
        public Optimizer getOptimizer() {
            return optimizer;
        }
    }

    public static class OptimizerGrad implements Gradient.OptimizerGrad {

        private Optimizer optimizer;

        public OptimizerGrad(Optimizer optimizer) {
            this.optimizer = optimizer;
        }

        @Override
        public Optimizer getOptimizer() {
            return optimizer;
        }

        @Override
        public PairList<String, NDArray> get() {
            return new PairList<>(
                    optimizer
                            .getParameters()
                            .stream()
                            .map(
                                    pair ->
                                            new Pair<>(
                                                    pair.getKey(),
                                                    ((MxNDArray) pair.getValue().getArray())
                                                            .getGradient()))
                            .collect(Collectors.toList()));
        }
    }

    public static class BlockKey implements Gradient.BlockKey {

        private Block block;

        public BlockKey(Block block) {
            this.block = block;
        }

        @Override
        public Block getBlock() {
            return block;
        }
    }

    public static class BlockGrad implements Gradient.BlockGrad {

        private Block block;

        public BlockGrad(Block block) {
            this.block = block;
        }

        @Override
        public Block getBlock() {
            return block;
        }

        @Override
        public PairList<String, NDArray> get() {
            return new PairList<>(
                    block.getParameters()
                            .stream()
                            .map(
                                    pair ->
                                            new Pair<>(
                                                    pair.getKey(),
                                                    ((MxNDArray) pair.getValue().getArray())
                                                            .getGradient()))
                            .collect(Collectors.toList()));
        }
    }

    public static class ParameterKey implements Gradient.ParameterKey {

        private Parameter parameter;

        public ParameterKey(Parameter parameter) {
            this.parameter = parameter;
        }

        @Override
        public Parameter getParameter() {
            return parameter;
        }
    }

    public static class ParameterGrad implements Gradient.ParameterGrad {

        private Parameter parameter;

        public ParameterGrad(Parameter parameter) {
            this.parameter = parameter;
        }

        @Override
        public Parameter getParameter() {
            return parameter;
        }

        @Override
        public NDArray get() {
            return ((MxNDArray) parameter.getArray()).getGradient();
        }
    }

    public static class NDArrayKey implements Gradient.NDArrayKey {

        private NDArray array;

        public NDArrayKey(NDArray array) {
            this.array = array;
        }

        @Override
        public NDArray getArray() {
            return array;
        }
    }

    public static class NDArrayGrad implements Gradient.NDArrayGrad {

        private NDArray array;

        public NDArrayGrad(NDArray array) {
            this.array = array;
        }

        @Override
        public NDArray getArray() {
            return array;
        }

        @Override
        public NDArray get() {
            return ((MxNDArray) array).getGradient();
        }
    }
}
