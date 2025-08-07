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
public class Source {

    private String type;

    @SerializedName("media_type")
    private String mediaType;

    private String data;
    private String url;

    Source(Builder builder) {
        this.type = builder.type;
        this.mediaType = builder.mediaType;
        this.data = builder.data;
        this.url = builder.url;
    }

    public String getType() {
        return type;
    }

    public String getMediaType() {
        return mediaType;
    }

    public String getData() {
        return data;
    }

    public String getUrl() {
        return url;
    }

    /**
     * Creates a builder to build a {@code Source}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code Source}. */
    public static final class Builder {

        String type;
        String mediaType;
        String data;
        String url;

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder mediaType(String mediaType) {
            this.mediaType = mediaType;
            return this;
        }

        public Builder data(String data) {
            this.data = data;
            return this;
        }

        public Builder url(String url) {
            this.url = url;
            return this;
        }

        public Source build() {
            return new Source(this);
        }
    }
}
