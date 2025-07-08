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
public class Usage {

    @SerializedName("completion_tokens")
    private int completionTokens;

    @SerializedName("prompt_tokens")
    private int promptTokens;

    @SerializedName("total_tokens")
    private int totalTokens;

    public Usage(int completionTokens, int promptTokens, int totalTokens) {
        this.completionTokens = completionTokens;
        this.promptTokens = promptTokens;
        this.totalTokens = totalTokens;
    }

    public int getCompletionTokens() {
        return completionTokens;
    }

    public int getPromptTokens() {
        return promptTokens;
    }

    public int getTotalTokens() {
        return totalTokens;
    }
}
