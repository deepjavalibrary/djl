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

/** A class that holds error message. */
public class ErrorResponse {

    private int code;
    private String type;
    private String message;

    /**
     * Constructs a {@code ErrorResponse} instance with code, type and message.
     *
     * @param code the error code
     * @param type the error type
     * @param message the error message
     */
    public ErrorResponse(int code, String type, String message) {
        this.code = code;
        this.type = type;
        this.message = message;
    }

    /**
     * Returns the error code.
     *
     * @return the error code
     */
    public int getCode() {
        return code;
    }

    /**
     * Returns the error type.
     *
     * @return the error type
     */
    public String getType() {
        return type;
    }

    /**
     * Returns the error message.
     *
     * @return the error message
     */
    public String getMessage() {
        return message;
    }
}
