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

import java.util.List;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class FunctionCallingConfig {

    private List<String> allowedFunctionNames;
    private FunctionCallingConfigMode mode;

    FunctionCallingConfig(Builder builder) {
        allowedFunctionNames = builder.allowedFunctionNames;
        mode = builder.mode;
    }

    public List<String> getAllowedFunctionNames() {
        return allowedFunctionNames;
    }

    public FunctionCallingConfigMode getMode() {
        return mode;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code FunctionCallingConfig}. */
    public static final class Builder {

        List<String> allowedFunctionNames;
        FunctionCallingConfigMode mode;

        public Builder allowedFunctionNames(List<String> allowedFunctionNames) {
            this.allowedFunctionNames = allowedFunctionNames;
            return this;
        }

        public Builder mode(FunctionCallingConfigMode mode) {
            this.mode = mode;
            return this;
        }

        public FunctionCallingConfig build() {
            return new FunctionCallingConfig(this);
        }
    }
}
