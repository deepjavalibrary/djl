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
package ai.djl.genai.openai;

import com.google.gson.annotations.SerializedName;

/** A data class represents chat completion schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Choice {

    private int index;
    private Message message;
    private Logprobs logprobs;
    private Message delta;

    @SerializedName("finish_reason")
    private String finishReason;

    public Choice(
            int index, Message message, Logprobs logprobs, Message delta, String finishReason) {
        this.index = index;
        this.message = message;
        this.logprobs = logprobs;
        this.delta = delta;
        this.finishReason = finishReason;
    }

    public int getIndex() {
        return index;
    }

    public Message getMessage() {
        return message;
    }

    public Message getDelta() {
        return delta;
    }

    public Logprobs getLogprobs() {
        return logprobs;
    }

    public String getFinishReason() {
        return finishReason;
    }
}
