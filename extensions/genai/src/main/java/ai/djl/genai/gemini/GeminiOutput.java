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

import ai.djl.genai.gemini.types.Candidate;
import ai.djl.genai.gemini.types.Content;
import ai.djl.genai.gemini.types.Part;
import ai.djl.genai.gemini.types.UsageMetadata;

import java.util.Collections;
import java.util.List;

/** A class presents the Gemini input. */
public class GeminiOutput {

    private List<Candidate> candidates;
    private UsageMetadata usageMetadata;
    private String modelVersion;

    GeminiOutput(List<Candidate> candidates, UsageMetadata usageMetadata, String modelVersion) {
        this.candidates = candidates;
        this.usageMetadata = usageMetadata;
        this.modelVersion = modelVersion;
    }

    /**
     * Returns the candidates.
     *
     * @return the candidates
     */
    public List<Candidate> getCandidates() {
        if (candidates == null) {
            return Collections.emptyList();
        }
        return candidates;
    }

    /**
     * Returns the first candidate if there is any.
     *
     * @return the candidate
     */
    public Candidate getCandidate() {
        if (candidates == null || candidates.isEmpty()) {
            return null;
        }
        return candidates.get(0);
    }

    /**
     * Returns the usage metadata.
     *
     * @return the usage metadata
     */
    public UsageMetadata getUsageMetadata() {
        return usageMetadata;
    }

    /**
     * Returns the model version.
     *
     * @return the model version
     */
    public String getModelVersion() {
        return modelVersion;
    }

    /**
     * Returns the aggregated text output.
     *
     * @return the aggregated text output
     */
    public String getTextOutput() {
        if (candidates == null || candidates.isEmpty()) {
            return "";
        }
        Content content = candidates.get(0).getContent();
        List<Part> parts = content.getParts();
        if (parts == null || parts.isEmpty()) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        for (Part part : content.getParts()) {
            if (part.getText() != null) {
                sb.append(part.getText());
            }
        }
        return sb.toString();
    }
}
