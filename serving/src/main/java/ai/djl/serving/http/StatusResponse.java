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

/** A class that holds model server status. */
public class StatusResponse {

    private String status;

    /** Constructs a new {@code StatusResponse} instance. */
    public StatusResponse() {}

    /**
     * Constructs a new {@code StatusResponse} instance with status line.
     *
     * @param status the status line
     */
    public StatusResponse(String status) {
        this.status = status;
    }

    /**
     * Returns the status.
     *
     * @return the status
     */
    public String getStatus() {
        return status;
    }
}
