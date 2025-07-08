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
package ai.djl.genai.gemini.types;

import java.util.Base64;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class SearchEntryPoint {

    private String renderedContent;
    private String sdkBlob;

    SearchEntryPoint(Builder builder) {
        renderedContent = builder.renderedContent;
        sdkBlob = builder.sdkBlob;
    }

    public String getRenderedContent() {
        return renderedContent;
    }

    public String getSdkBlob() {
        return sdkBlob;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code SearchEntryPoint}. */
    public static final class Builder {

        String renderedContent;
        String sdkBlob;

        public Builder renderedContent(String renderedContent) {
            this.renderedContent = renderedContent;
            return this;
        }

        public Builder sdkBlob(byte[] sdkBlob) {
            this.sdkBlob = Base64.getEncoder().encodeToString(sdkBlob);
            return this;
        }

        public SearchEntryPoint build() {
            return new SearchEntryPoint(this);
        }
    }
}
