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
package ai.djl.modality;

/** A class stores the generic inference results. */
public class Output extends Input {

    private int code;
    private String message;

    /** Constructs a {@code Output} instance. */
    public Output() {
        this(200, "OK");
    }

    /**
     * Constructs a {@code Output} with specified {@code requestId}, {@code code} and {@code
     * message}.
     *
     * @param code the status code of the output
     * @param message the status message of the output
     */
    public Output(int code, String message) {
        this.code = code;
        this.message = message;
    }

    /**
     * Returns the status code of the output.
     *
     * @return the status code of the output
     */
    public int getCode() {
        return code;
    }

    /**
     * Sets the status code of the output.
     *
     * @param code the status code of the output
     */
    public void setCode(int code) {
        this.code = code;
    }

    /**
     * Returns the status code of the output.
     *
     * @return the status code of the output
     */
    public String getMessage() {
        return message;
    }

    /**
     * Sets the status message of the output.
     *
     * @param message the status message of the output
     */
    public void setMessage(String message) {
        this.message = message;
    }
}
