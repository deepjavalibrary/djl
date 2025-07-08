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
public class MultiSpeakerVoiceConfig {

    private List<SpeakerVoiceConfig> speakerVoiceConfigs;

    MultiSpeakerVoiceConfig(Builder builder) {
        speakerVoiceConfigs = builder.speakerVoiceConfigs;
    }

    public List<SpeakerVoiceConfig> getSpeakerVoiceConfigs() {
        return speakerVoiceConfigs;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code MultiSpeakerVoiceConfig}. */
    public static final class Builder {

        List<SpeakerVoiceConfig> speakerVoiceConfigs = new ArrayList<>();

        public Builder speakerVoiceConfigs(List<SpeakerVoiceConfig> speakerVoiceConfigs) {
            this.speakerVoiceConfigs.clear();
            this.speakerVoiceConfigs.addAll(speakerVoiceConfigs);
            return this;
        }

        public Builder addSpeakerVoiceConfig(SpeakerVoiceConfig speakerVoiceConfig) {
            this.speakerVoiceConfigs.add(speakerVoiceConfig);
            return this;
        }

        public Builder addSpeakerVoiceConfig(SpeakerVoiceConfig.Builder speakerVoiceConfig) {
            this.speakerVoiceConfigs.add(speakerVoiceConfig.build());
            return this;
        }

        public MultiSpeakerVoiceConfig build() {
            return new MultiSpeakerVoiceConfig(this);
        }
    }
}
