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
public class MessageStart {

    private String id;
    private String model;

    @SerializedName("stop_reason")
    private String stopReason;

    @SerializedName("stop_sequence")
    private String stopSequence;

    private Usage usage;

    public MessageStart(
            String id, String model, String stopReason, String stopSequence, Usage usage) {
        this.id = id;
        this.model = model;
        this.stopReason = stopReason;
        this.stopSequence = stopSequence;
        this.usage = usage;
    }

    public String getId() {
        return id;
    }

    public String getModel() {
        return model;
    }

    public String getStopReason() {
        return stopReason;
    }

    public String getStopSequence() {
        return stopSequence;
    }

    public Usage getUsage() {
        return usage;
    }
}
