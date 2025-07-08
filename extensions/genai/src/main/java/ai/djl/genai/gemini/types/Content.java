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

import java.util.ArrayList;
import java.util.List;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Content {

    private List<Part> parts;
    private String role;

    Content(Builder builder) {
        parts = builder.parts;
        role = builder.role;
    }

    public List<Part> getParts() {
        return parts;
    }

    public String getRole() {
        return role;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static Builder text(String text) {
        return builder().addPart(Part.text(text)).role("user");
    }

    public static Builder fileData(String fileUri, String mimeType) {
        return builder().addPart(Part.fileData(fileUri, mimeType)).role("user");
    }

    public static Builder inlineData(byte[] bytes, String mimeType) {
        return builder().addPart(Part.inlineData(bytes, mimeType)).role("user");
    }

    /** Builder class for {@code Content}. */
    public static final class Builder {

        List<Part> parts = new ArrayList<>();
        String role;

        public Builder parts(List<Part> parts) {
            this.parts.clear();
            this.parts.addAll(parts);
            return this;
        }

        public Builder addPart(Part part) {
            this.parts.add(part);
            return this;
        }

        public Builder addPart(Part.Builder part) {
            this.parts.add(part.build());
            return this;
        }

        public Builder role(String role) {
            this.role = role;
            return this;
        }

        public Content build() {
            return new Content(this);
        }
    }
}
