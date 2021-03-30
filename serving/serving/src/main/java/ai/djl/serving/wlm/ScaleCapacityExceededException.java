/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.wlm;

/**
 * Is thrown when capacity of workers is reached during autoscaling.
 *
 * @author erik.bamberg@web.de
 */
public class ScaleCapacityExceededException extends Exception {

    /** serialVersionUDI for this class cause exceptions are serializable. */
    private static final long serialVersionUID = 1633130362838844091L;

    /** No arguments. */
    public ScaleCapacityExceededException() {}

    /**
     * construct using a message.
     *
     * @param message the message.
     */
    public ScaleCapacityExceededException(String message) {
        super(message);
    }

    /**
     * construct using a cause.
     *
     * @param cause the root cause.
     */
    public ScaleCapacityExceededException(Throwable cause) {
        super(cause);
    }

    /**
     * construct using a message and a clause.
     *
     * @param message the message.
     * @param cause the root cause.
     */
    public ScaleCapacityExceededException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * construct using a message cause and flags.
     *
     * @param message the message.
     * @param cause the root cause.
     * @param enableSuppression enable suppression or not.
     * @param writableStackTrace flag if writableStackTrace.
     */
    public ScaleCapacityExceededException(
            String message,
            Throwable cause,
            boolean enableSuppression,
            boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
