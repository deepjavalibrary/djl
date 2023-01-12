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
package ai.djl.ndarray.refcount;

/** A runtime exception thrown within the reference counting package. */
public class RCException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs a new runtime exception with the specified detail message.
     *
     * @param message - the message to be displayed
     */
    public RCException(String message) {
        super(message);
    }

    /**
     * Constructs a new runtime exception with the specified detail message and cause.
     *
     * @param message - the message to be displayed
     * @param cause - the cause of the exception
     */
    public RCException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * Constructs a new runtime exception with the specified cause.
     *
     * @param cause - the cause of the exception
     */
    public RCException(Throwable cause) {
        super(cause);
    }
}
