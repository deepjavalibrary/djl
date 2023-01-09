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
package ai.djl.ndarray.gc;

import ai.djl.ndarray.NDArray;

/**
 * {@code GCRuntimeException} is the exception thrown when the {@link DynamicInvocationHandler}
 * fails to collect the {@link NDArray} object or call a method on the {@link NDArray} object.
 */
public class GCRuntimeException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates a new instance of {@code GCRuntimeException}.
     *
     * @param message the message
     */
    public GCRuntimeException(String message) {
        super(message);
    }

    /**
     * Creates a new instance of {@code GCRuntimeException}.
     *
     * @param e the exception
     */
    public GCRuntimeException(Exception e) {
        super(e);
    }
}
