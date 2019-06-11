package org.apache.mxnet.engine;

import org.apache.mxnet.jna.JnaUtils;

public final class MxAutograd {

    private MxAutograd() {
        // not callable
    }

    /**
     * Get status on whether Autograd is recording computations
     *
     * @return Current state of recording.
     */
    public static boolean isRecording() {
        return JnaUtils.autogradIsRecording();
    }

    /**
     * Get status on whether Autograd is in training/predicting mode
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
     * @param isTraining true if for training
     * @return the previous status before this set
     */
    public static boolean setTraining(boolean isTraining) {
        return JnaUtils.autogradSetTraining(isTraining);
    }
}
