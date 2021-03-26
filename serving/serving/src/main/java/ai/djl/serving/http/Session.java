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

import io.netty.handler.codec.http.HttpRequest;
import java.util.UUID;

/** A class that holds HTTP session information. */
public class Session {

    private String requestId;
    private String remoteIp;
    private String method;
    private String uri;
    private String protocol;
    private int code;
    private long startTime;

    /**
     * Constructs a new {@code Session} instance.
     *
     * @param remoteIp the remote IP address
     * @param request the HTTP request
     */
    public Session(String remoteIp, HttpRequest request) {
        this.remoteIp = remoteIp;
        this.uri = request.uri();
        if (request.decoderResult().isSuccess()) {
            method = request.method().name();
            protocol = request.protocolVersion().text();
        } else {
            method = "GET";
            protocol = "HTTP/1.1";
        }
        requestId = UUID.randomUUID().toString();
        startTime = System.currentTimeMillis();
    }

    /**
     * Returns the request ID.
     *
     * @return the request ID
     */
    public String getRequestId() {
        return requestId;
    }

    /**
     * Sets the HTTP response code.
     *
     * @param code the HTTP response code
     */
    public void setCode(int code) {
        this.code = code;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        long duration = System.currentTimeMillis() - startTime;
        return remoteIp + " \"" + method + " " + uri + ' ' + protocol + "\" " + code + ' '
                + duration;
    }
}
