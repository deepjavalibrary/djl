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
package ai.djl.genai.gemini;

import ai.djl.util.JsonUtils;

import java.util.Iterator;

/** A stream version of {@link GeminiOutput}. */
public class StreamGeminiOutput implements Iterable<GeminiOutput> {

    private transient Iterator<String> output;

    StreamGeminiOutput(Iterator<String> output) {
        this.output = output;
    }

    /** {@inheritDoc} */
    @Override
    public Iterator<GeminiOutput> iterator() {
        return new Iterator<GeminiOutput>() {

            /** {@inheritDoc} */
            @Override
            public boolean hasNext() {
                return output.hasNext();
            }

            /** {@inheritDoc} */
            @Override
            public GeminiOutput next() {
                String json = output.next();
                if (json.isEmpty()) {
                    return new GeminiOutput(null, null, null);
                }
                return JsonUtils.GSON.fromJson(json, GeminiOutput.class);
            }
        };
    }

    /**
     * Customizes schema deserialization.
     *
     * @param output the output iterator
     * @return the deserialized {@code StreamGeminiOutput} instance
     */
    public static StreamGeminiOutput fromJson(Iterator<String> output) {
        return new StreamGeminiOutput(output);
    }
}
