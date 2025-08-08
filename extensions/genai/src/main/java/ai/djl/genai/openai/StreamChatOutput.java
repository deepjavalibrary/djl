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

import ai.djl.genai.anthropic.AnthropicOutput;
import ai.djl.genai.anthropic.StreamAnthropicOutput;
import ai.djl.util.JsonUtils;

import com.google.gson.JsonObject;

import java.util.Iterator;

/** A stream version of {@link ChatOutput}. */
public class StreamChatOutput implements Iterable<ChatOutput> {

    private transient Iterator<String> output;

    StreamChatOutput(Iterator<String> output) {
        this.output = output;
    }

    /** {@inheritDoc} */
    @Override
    public Iterator<ChatOutput> iterator() {
        return new Iterator<ChatOutput>() {

            /** {@inheritDoc} */
            @Override
            public boolean hasNext() {
                return output.hasNext();
            }

            /** {@inheritDoc} */
            @Override
            public ChatOutput next() {
                String json = output.next();
                if (json.isEmpty() || "[DONE]".equals(json)) {
                    return new ChatOutput();
                }
                JsonObject element = JsonUtils.GSON.fromJson(json, JsonObject.class);
                if (element.has("type")) {
                    AnthropicOutput.Builder builder = AnthropicOutput.builder();
                    StreamAnthropicOutput.next(builder, element, output);
                    AnthropicOutput ant = builder.build();
                    return ChatOutput.fromAnthropic(ant);
                }
                return ChatOutput.fromJson(json);
            }
        };
    }

    /**
     * Customizes schema deserialization.
     *
     * @param output the output iterator
     * @return the deserialized {@code StreamChatOutput} instance
     */
    public static StreamChatOutput fromJson(Iterator<String> output) {
        return new StreamChatOutput(output);
    }
}
