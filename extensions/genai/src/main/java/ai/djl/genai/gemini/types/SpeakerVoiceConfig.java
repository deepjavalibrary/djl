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
public class SpeakerVoiceConfig {

    private String speaker;
    private VoiceConfig voiceConfig;

    SpeakerVoiceConfig(Builder builder) {
        speaker = builder.speaker;
        voiceConfig = builder.voiceConfig;
    }

    public String getSpeaker() {
        return speaker;
    }

    public VoiceConfig getVoiceConfig() {
        return voiceConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code SpeakerVoiceConfig}. */
    public static final class Builder {

        String speaker;
        VoiceConfig voiceConfig;

        public Builder speaker(String speaker) {
            this.speaker = speaker;
            return this;
        }

        public Builder voiceConfig(VoiceConfig voiceConfig) {
            this.voiceConfig = voiceConfig;
            return this;
        }

        public Builder voiceConfig(VoiceConfig.Builder voiceConfig) {
            this.voiceConfig = voiceConfig.build();
            return this;
        }

        public SpeakerVoiceConfig build() {
            return new SpeakerVoiceConfig(this);
        }
    }
}
