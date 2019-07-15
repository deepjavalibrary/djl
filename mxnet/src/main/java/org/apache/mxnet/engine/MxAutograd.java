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

import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MxAutograd implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    public MxAutograd() {
        Boolean prevRecordingState = setRecording(true);
        if (prevRecordingState) {
            logger.warn(
                    "Autograd Recording is already set to True. "
                            + "Please use try with resource or call autograd.setRecording(false) after training.");
        }
        Boolean prevTrainingState = setTraining(true);
        if (prevTrainingState) {
            logger.warn(
                    "Autograd Training is already set to True. "
                            + "Please use try with resource or call autograd.setTraining(false) after training.");
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        setRecording(false);
        setTraining(false);
    }

    /**
     * Run backward and calculate gradient w.r.t previously marked variable (head).
     *
     * @param array target NDArray to run backward and calculate gradient w.r.t head
     */
    public void backward(MxNDArray array) {
        array.backward();
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
}
