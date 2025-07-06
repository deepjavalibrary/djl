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
package ai.djl.genai.huggingface;

import ai.djl.util.JsonUtils;

import java.util.Iterator;

/** A stream version of {@link GenerationOutput}. */
public class StreamGenerationOutput implements Iterable<GenerationOutput> {

    private transient Iterator<String> output;

    StreamGenerationOutput(Iterator<String> output) {
        this.output = output;
    }

    /** {@inheritDoc} */
    @Override
    public Iterator<GenerationOutput> iterator() {
        return new Iterator<GenerationOutput>() {

            /** {@inheritDoc} */
            @Override
            public boolean hasNext() {
                return output.hasNext();
            }

            /** {@inheritDoc} */
            @Override
            public GenerationOutput next() {
                String json = output.next();
                if (json.isEmpty()) {
                    return new GenerationOutput(null, null, null);
                }
                return JsonUtils.GSON.fromJson(json, GenerationOutput.class);
            }
        };
    }

    /**
     * Customizes schema deserialization.
     *
     * @param output the output iterator
     * @return the deserialized {@code StreamGenerationOutput} instance
     */
    public static StreamGenerationOutput fromJson(Iterator<String> output) {
        return new StreamGenerationOutput(output);
    }
}
