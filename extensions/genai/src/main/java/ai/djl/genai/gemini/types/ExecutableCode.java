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
public class ExecutableCode {

    private String code;
    private Language language;

    ExecutableCode(Builder builder) {
        code = builder.code;
        language = builder.language;
    }

    public String getCode() {
        return code;
    }

    public Language getLanguage() {
        return language;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code ExecutableCode}. */
    public static final class Builder {

        String code;
        Language language;

        public Builder code(String code) {
            this.code = code;
            return this;
        }

        public Builder language(Language language) {
            this.language = language;
            return this;
        }

        public ExecutableCode build() {
            return new ExecutableCode(this);
        }
    }
}
