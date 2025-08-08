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

import ai.djl.util.JsonUtils;

import com.google.gson.JsonObject;

import java.util.Collections;
import java.util.Iterator;

/** A stream version of {@link AnthropicOutput}. */
public class StreamAnthropicOutput implements Iterable<AnthropicOutput> {

    private transient Iterator<String> output;

    StreamAnthropicOutput(Iterator<String> output) {
        this.output = output;
    }

    /** {@inheritDoc} */
    @Override
    public Iterator<AnthropicOutput> iterator() {
        return new Iterator<AnthropicOutput>() {

            /** {@inheritDoc} */
            @Override
            public boolean hasNext() {
                return output.hasNext();
            }

            /** {@inheritDoc} */
            @Override
            public AnthropicOutput next() {
                AnthropicOutput.Builder builder = AnthropicOutput.builder();
                StreamAnthropicOutput.next(builder, output.next(), output);
                return builder.build();
            }
        };
    }

    static void next(AnthropicOutput.Builder builder, String json, Iterator<String> output) {
        if (json.isEmpty()) {
            next(builder, output.next(), output);
            return;
        }
        JsonObject element = JsonUtils.GSON.fromJson(json, JsonObject.class);
        next(builder, element, output);
    }

    /**
     * Processes streaming output.
     *
     * @param builder the builder
     * @param element the json stream chunk
     * @param output the iterator
     */
    public static void next(
            AnthropicOutput.Builder builder, JsonObject element, Iterator<String> output) {
        String type = element.get("type").getAsString();
        if ("ping".equals(type)
                || "content_block_start".equals(type)
                || "content_block_stop".equals(type)) {
            next(builder, output.next(), output);
        } else if ("message_start".equals(type)) {
            MessageStart message =
                    JsonUtils.GSON.fromJson(element.get("message"), MessageStart.class);
            builder.id(message.getId())
                    .model(message.getModel())
                    .usage(message.getUsage())
                    .stopReason(message.getStopReason())
                    .stopSequence(message.getStopSequence());
            next(builder, output.next(), output);
        } else if ("content_block_delta".equals(type)) {
            int index = element.get("index").getAsInt();
            ContentBlockDelta delta =
                    JsonUtils.GSON.fromJson(element.get("delta"), ContentBlockDelta.class);
            builder.content(
                    Collections.singletonList(Content.text(delta.getText()).index(index).build()));
        } else if ("message_delta".equals(type)) {
            MessageStart message =
                    JsonUtils.GSON.fromJson(element.get("delta"), MessageStart.class);
            builder.usage(message.getUsage())
                    .stopReason(message.getStopReason())
                    .stopSequence(message.getStopSequence());
            next(builder, output.next(), output);
        } else if ("message_stop".equals(type)) {
            builder.content(Collections.singletonList(Content.text("").build()));
        } else {
            throw new IllegalArgumentException("Unexpected event: " + type);
        }
    }

    /**
     * Customizes schema deserialization.
     *
     * @param output the output iterator
     * @return the deserialized {@code StreamAnthropicOutput} instance
     */
    public static StreamAnthropicOutput fromJson(Iterator<String> output) {
        return new StreamAnthropicOutput(output);
    }
}
