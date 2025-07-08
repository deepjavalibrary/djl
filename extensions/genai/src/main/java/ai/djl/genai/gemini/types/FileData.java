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

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class FileData {

    private String displayName;
    private String fileUri;
    private String mimeType;

    FileData(Builder builder) {
        displayName = builder.displayName;
        fileUri = builder.fileUri;
        mimeType = builder.mimeType;
    }

    public String getDisplayName() {
        return displayName;
    }

    public String getFileUri() {
        return fileUri;
    }

    public String getMimeType() {
        return mimeType;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code FileData}. */
    public static final class Builder {

        String displayName;
        String fileUri;
        String mimeType;

        public Builder displayName(String displayName) {
            this.displayName = displayName;
            return this;
        }

        public Builder fileUri(String fileUri) {
            this.fileUri = fileUri;
            return this;
        }

        public Builder mimeType(String mimeType) {
            this.mimeType = mimeType;
            return this;
        }

        public FileData build() {
            return new FileData(this);
        }
    }
}
