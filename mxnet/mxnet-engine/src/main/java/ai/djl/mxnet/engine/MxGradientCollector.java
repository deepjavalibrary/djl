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
package ai.djl.mxnet.engine;

import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;

/** {@code MxGradientCollector} is the MXNet implementation of {@link GradientCollector}. */
public class MxGradientCollector implements GradientCollector {

    /**
     * Constructs an {@code MxGradientCollector} and enables training data collection for
     * backpropogation.
     */
    MxGradientCollector() {
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

    /**
     * Gets whether Autograd is recording computations.
     *
     * @return the current state of recording
     */
    public static boolean isRecording() {
        return JnaUtils.autogradIsRecording();
    }

    /**
     * Gets whether Autograd is in training/predicting mode.
     *
     * @return the current state of training/predicting
     */
    public static boolean isTraining() {
        return JnaUtils.autogradIsTraining();
    }

    /**
     * Sets the status to recording/not recording. When recording, graph will be constructed for
     * gradient computation.
     *
     * @param isRecording the recording state to be set
     * @return the previous recording state before this set
     */
    public static boolean setRecording(boolean isRecording) {
        return JnaUtils.autogradSetIsRecording(isRecording);
    }

    /**
     * Sets the status to training/predicting. This affects ctx.is_train in the device running the
     * operator. For example, Dropout will drop inputs randomly when isTraining=True, while simply
     * passing through if isTraining=False.
     *
     * @param isTraining {@code true} if for training
     * @return the previous status before this set
     */
    public static boolean setTraining(boolean isTraining) {
        return JnaUtils.autogradSetTraining(isTraining);
    }

    /**
     * Returns the {@link Symbol} of a network formed by the recorded operations on the given {@link
     * NDArray}.
     *
     * @param manager the {@link NDManager} to create the {@link Symbol}
     * @param array the {@link NDArray}
     * @return the {@link Symbol}
     */
    public static Symbol getSymbol(NDManager manager, NDArray array) {
        return new Symbol((MxNDManager) manager, JnaUtils.autogradGetSymbol(array));
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        setRecording(false);
        setTraining(false);
    }

    /** {@inheritDoc} */
    @Override
    public void backward(NDArray array) {
        backward(array, false);
    }

    /**
     * Computes the gradients of the NDArray w.r.t variables.
     *
     * @param array the target/head array to run backward on
     * @param retainGraph whether to retain the computation graph for another backward pass on the
     *     same graph. By default the computation history is cleared.
     */
    private void backward(NDArray array, boolean retainGraph) {
        JnaUtils.autogradBackward(new NDList(array), retainGraph ? 1 : 0);
    }
}
