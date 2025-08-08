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
public class Metadata {

    @SerializedName("user_id")
    private String userId;

    Metadata(Builder builder) {
        this.userId = builder.userId;
    }

    public String getUserId() {
        return userId;
    }

    /**
     * Creates a builder to build a {@code Metadata}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code Metadata}. */
    public static final class Builder {

        String userId;

        public Builder userId(String userId) {
            this.userId = userId;
            return this;
        }

        public Metadata build() {
            return new Metadata(this);
        }
    }
}
