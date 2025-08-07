/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.genai.anthropic;

import com.google.gson.annotations.SerializedName;

/** A data class represents Anthropic schema. */
@SuppressWarnings("MissingJavadocMethod")
public class ServerToolUse {

    @SerializedName("web_search_requests")
    private int webSearchRequests;

    ServerToolUse(Builder builder) {
        this.webSearchRequests = builder.webSearchRequests;
    }

    public int getWebSearchRequests() {
        return webSearchRequests;
    }

    /**
     * Creates a builder to build a {@code ServerToolUse}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code ServerToolUse}. */
    public static final class Builder {

        int webSearchRequests;

        public Builder webSearchRequests(int webSearchRequests) {
            this.webSearchRequests = webSearchRequests;
            return this;
        }

        public ServerToolUse build() {
            return new ServerToolUse(this);
        }
    }
}
