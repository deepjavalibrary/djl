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

/** A data class represents Anthropic schema. */
@SuppressWarnings("MissingJavadocMethod")
public class CacheControl {

    private String type;
    private String ttl;

    CacheControl(Builder builder) {
        this.type = builder.type;
        this.ttl = builder.ttl;
    }

    public String getType() {
        return type;
    }

    public String getTtl() {
        return ttl;
    }

    /**
     * Creates a builder to build a {@code CacheControl}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code CacheControl}. */
    public static final class Builder {

        String type;
        String ttl;

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder ttl(String ttl) {
            this.ttl = ttl;
            return this;
        }

        public CacheControl build() {
            return new CacheControl(this);
        }
    }
}
