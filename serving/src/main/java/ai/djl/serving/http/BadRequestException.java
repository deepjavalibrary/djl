/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.http;

/** Thrown when a bad HTTP request is received. */
public class BadRequestException extends IllegalArgumentException {

    static final long serialVersionUID = 1L;

    /**
     * Constructs an {@code BadRequestException} with the specified detail message.
     *
     * @param message The detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method)
     */
    public BadRequestException(String message) {
        super(message);
    }

    /**
     * Constructs an {@code BadRequestException} with the specified detail message and a root cause.
     *
     * @param message The detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method)
     * @param cause root cause
     */
    public BadRequestException(String message, Throwable cause) {
        super(message, cause);
    }
}
