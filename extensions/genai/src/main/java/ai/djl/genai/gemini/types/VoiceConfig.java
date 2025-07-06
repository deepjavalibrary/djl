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
public class VoiceConfig {

    private PrebuiltVoiceConfig prebuiltVoiceConfig;

    VoiceConfig(Builder builder) {
        prebuiltVoiceConfig = builder.prebuiltVoiceConfig;
    }

    public PrebuiltVoiceConfig getPrebuiltVoiceConfig() {
        return prebuiltVoiceConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code VoiceConfig}. */
    public static final class Builder {

        PrebuiltVoiceConfig prebuiltVoiceConfig;

        public Builder prebuiltVoiceConfig(PrebuiltVoiceConfig prebuiltVoiceConfig) {
            this.prebuiltVoiceConfig = prebuiltVoiceConfig;
            return this;
        }

        public Builder prebuiltVoiceConfig(PrebuiltVoiceConfig.Builder prebuiltVoiceConfig) {
            this.prebuiltVoiceConfig = prebuiltVoiceConfig.build();
            return this;
        }

        public VoiceConfig build() {
            return new VoiceConfig(this);
        }
    }
}
