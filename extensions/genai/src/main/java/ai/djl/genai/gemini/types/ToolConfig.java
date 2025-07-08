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
public class ToolConfig {

    private FunctionCallingConfig functionCallingConfig;
    private RetrievalConfig retrievalConfig;

    ToolConfig(Builder builder) {
        functionCallingConfig = builder.functionCallingConfig;
        retrievalConfig = builder.retrievalConfig;
    }

    public FunctionCallingConfig getFunctionCallingConfig() {
        return functionCallingConfig;
    }

    public RetrievalConfig getRetrievalConfig() {
        return retrievalConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code ToolConfig}. */
    public static final class Builder {

        FunctionCallingConfig functionCallingConfig;
        RetrievalConfig retrievalConfig;

        public Builder functionCallingConfig(FunctionCallingConfig functionCallingConfig) {
            this.functionCallingConfig = functionCallingConfig;
            return this;
        }

        public Builder functionCallingConfig(FunctionCallingConfig.Builder functionCallingConfig) {
            this.functionCallingConfig = functionCallingConfig.build();
            return this;
        }

        public Builder retrievalConfig(RetrievalConfig retrievalConfig) {
            this.retrievalConfig = retrievalConfig;
            return this;
        }

        public Builder retrievalConfig(RetrievalConfig.Builder retrievalConfig) {
            this.retrievalConfig = retrievalConfig.build();
            return this;
        }

        public ToolConfig build() {
            return new ToolConfig(this);
        }
    }
}
