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
public class Container {

    @SerializedName("expires_at")
    private String expiresAt;

    private String id;

    Container(Builder builder) {
        this.expiresAt = builder.expiresAt;
        this.id = builder.id;
    }

    public String getExpiresAt() {
        return expiresAt;
    }

    public String getId() {
        return id;
    }

    /**
     * Creates a builder to build a {@code Container}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code Container}. */
    public static final class Builder {

        String expiresAt;
        String id;

        public Builder expiresAt(String expiresAt) {
            this.expiresAt = expiresAt;
            return this;
        }

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Container build() {
            return new Container(this);
        }
    }
}
