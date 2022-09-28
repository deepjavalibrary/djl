/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.nn;

/** Thrown to indicate that a {@link Parameter} was not initialized. */
public class UninitializeParameterException extends RuntimeException {

    private static final long serialVersionUID = 1L;
    private final Parameter parameter;

    /**
     * Constructs a new exception with the specified detail message. The cause is not initialized,
     * and may subsequently be initialized by a call to {@link #initCause}.
     *
     * @param parameter the parameter that was not initialized
     * @param message the detail message that is saved for later retrieval by the {@link
     *     #getMessage()} method
     */
    public UninitializeParameterException(Parameter parameter, String message) {
        super(message);
        this.parameter = parameter;
    }

    /**
     * Constructs a new exception with the specified detail message and cause.
     *
     * <p>Note that the detail message associated with {@code cause} is <i>not</i> automatically
     * incorporated in this exception's detail message.
     *
     * @param parameter the parameter that was not initialized
     * @param message the detail message that is saved for later retrieval by the {@link
     *     #getMessage()} method
     * @param cause the cause that is saved for later retrieval by the {@link #getCause()} method. A
     *     {@code null} value is permitted, and indicates that the cause is nonexistent or unknown
     */
    public UninitializeParameterException(Parameter parameter, String message, Throwable cause) {
        super(message, cause);
        this.parameter = parameter;
    }

    /**
     * Constructs a new exception with the specified cause and a detail message of {@code
     * (cause==null ? null : cause.toString())} which typically contains the class and detail
     * message of {@code cause}. This constructor is useful for exceptions that are little more than
     * wrappers for other throwables. For example, {@link java.security.PrivilegedActionException}.
     *
     * @param parameter the parameter that was not initialized
     * @param cause the cause that is saved for later retrieval by the {@link #getCause()} method. A
     *     {@code null} value is permitted, and indicates that the cause is nonexistent or unknown
     */
    public UninitializeParameterException(Parameter parameter, Throwable cause) {
        super(cause);
        this.parameter = parameter;
    }

    /**
     * Returns the parameter that was not initialized.
     *
     * @return the parameter that was not initialized
     */
    public Parameter getParameter() {
        return parameter;
    }
}
