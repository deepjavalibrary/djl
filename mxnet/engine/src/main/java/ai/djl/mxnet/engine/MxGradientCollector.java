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
import com.sun.jna.Pointer;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.training.GradientCollector;

public class MxGradientCollector implements GradientCollector {

    public MxGradientCollector() {
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
     * Set status to training/predicting. This affects ctx.is_train in operator running device. For
     * example, Dropout will drop inputs randomly when isTraining=True while simply passing through
     * if isTraining=False.
     *
     * @param isTraining {@code true} if for training
     * @return the previous status before this set
     */
    public static boolean setTraining(boolean isTraining) {
        return JnaUtils.autogradSetTraining(isTraining);
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
        backward(array, null, false, true);
    }

    /**
     * Computes the gradients of the NDArray w.r.t variables.
     *
     * @param array target/head array to run backward on.
     * @param outGrad output gradient NDArray
     * @param retainGraph Whether to retain the computation graph for another backward pass on the
     *     same graph. By default the computation history is cleared.
     * @param isTraining Whether to compute gradient for training or inference.
     */
    private void backward(NDArray array, NDArray outGrad, boolean retainGraph, boolean isTraining) {
        Pointer outGradHandle;
        if (outGrad != null) {
            MxNDArray outGradND = (MxNDArray) outGrad;
            outGradHandle = outGradND.getHandle();
        } else {
            outGradHandle = null;
        }

        JnaUtils.autogradBackwardExecute(
                1,
                ((MxNDArray) array).getHandle(),
                outGradHandle,
                0,
                null,
                retainGraph ? 1 : 0,
                0,
                isTraining ? 1 : 0,
                null,
                null);
    }
}
