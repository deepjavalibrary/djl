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
import ai.djl.genai.gemini.types.FunctionCall;
import ai.djl.genai.gemini.types.LogprobsResult;
import ai.djl.genai.gemini.types.LogprobsResultCandidate;
import ai.djl.genai.gemini.types.LogprobsResultTopCandidates;
import ai.djl.genai.gemini.types.Part;
import ai.djl.genai.gemini.types.UsageMetadata;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** A class presents the Gemini input. */
public class GeminiOutput {

    private List<Candidate> candidates;
    private UsageMetadata usageMetadata;
    private String modelVersion;
    private String createTime;
    private String responseId;

    GeminiOutput(
            List<Candidate> candidates,
            UsageMetadata usageMetadata,
            String modelVersion,
            String createTime,
            String responseId) {
        this.candidates = candidates;
        this.usageMetadata = usageMetadata;
        this.modelVersion = modelVersion;
        this.createTime = createTime;
        this.responseId = responseId;
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
     * Returns the per token log probability and the alternative tokens.
     *
     * @return the per token log probability and the alternative tokens.
     */
    public PairList<LogprobsResultCandidate, List<LogprobsResultCandidate>> getLogprobsResult() {
        Candidate candidate = getCandidate();
        if (candidate == null) {
            return null;
        }
        LogprobsResult result = candidate.getLogprobsResult();
        if (result == null) {
            return null;
        }
        PairList<LogprobsResultCandidate, List<LogprobsResultCandidate>> pairs = new PairList<>();
        List<LogprobsResultTopCandidates> top = result.getTopCandidates();
        int i = 0;
        for (LogprobsResultCandidate lr : result.getChosenCandidates()) {
            List<LogprobsResultCandidate> alternatives = new ArrayList<>();
            if (top != null && i < top.size()) {
                String token = lr.getToken();
                for (LogprobsResultCandidate alt : top.get(i).getCandidates()) {
                    if (!alt.getToken().equals(token)) {
                        alternatives.add(alt);
                    }
                }
            }
            pairs.add(lr, alternatives);
            ++i;
        }
        return pairs;
    }

    /**
     * Returns the content parts.
     *
     * @return the content parts
     */
    public List<Part> getParts() {
        Candidate candidate = getCandidate();
        if (candidate == null) {
            return Collections.emptyList();
        }
        List<Part> parts = candidate.getContent().getParts();
        if (parts == null) {
            return Collections.emptyList();
        }
        return parts;
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
     * Returns the creation time.
     *
     * @return the creation time
     */
    public String getCreateTime() {
        return createTime;
    }

    /**
     * Returns the response id.
     *
     * @return the response id
     */
    public String getResponseId() {
        return responseId;
    }

    /**
     * Returns the aggregated text output.
     *
     * @return the aggregated text output
     */
    public String getTextOutput() {
        StringBuilder sb = new StringBuilder();
        for (Part part : getParts()) {
            if (part.getText() != null) {
                sb.append(part.getText());
            }
        }
        return sb.toString();
    }

    /**
     * Returns the {@link FunctionCall}s in the response.
     *
     * @return the {@link FunctionCall}s in the response
     */
    public List<FunctionCall> getFunctionCalls() {
        List<FunctionCall> list = new ArrayList<>();
        for (Part part : getParts()) {
            FunctionCall call = part.getFunctionCall();
            if (call != null) {
                list.add(call);
            }
        }
        return list;
    }

    /**
     * Returns the first {@link FunctionCall} in the response.
     *
     * @return the first {@link FunctionCall} in the response
     */
    public FunctionCall getFunctionCall() {
        List<FunctionCall> list = getFunctionCalls();
        if (list.isEmpty()) {
            return null;
        }
        return list.get(0);
    }
}
