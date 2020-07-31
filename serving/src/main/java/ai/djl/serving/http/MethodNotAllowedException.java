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

/** Thrown when a HTTP request which method is not allowed. */
public class MethodNotAllowedException extends RuntimeException {

    static final long serialVersionUID = 1L;

    /**
     * Constructs an {@code MethodNotAllowedException} with {@code null} as its error detail
     * message.
     */
    public MethodNotAllowedException() {
        super("Requested method is not allowed, please refer to API document.");
    }
}
