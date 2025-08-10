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

import ai.djl.util.Utils;

/** Utility class to hold the Gemini models url. */
public final class Anthropic {

    public static final Anthropic OPUS_4_1 = new Anthropic("claude-opus-4-1@20250805");
    public static final Anthropic OPUS_4 = new Anthropic("claude-opus-4@20250514");
    public static final Anthropic SONNET_4 = new Anthropic("claude-sonnet-4@20250514");
    public static final Anthropic SONNET_3_7 = new Anthropic("claude-3-7-sonnet@20250219");
    public static final Anthropic SONNET_3_5 = new Anthropic("claude-3-5-sonnet@20240620");
    public static final Anthropic HAIKU_3_5 = new Anthropic("claude-3-5-haiku@20241022");
    public static final Anthropic HAIKU_3 = new Anthropic("claude-3-haiku@20240307");

    private String model;

    Anthropic(String model) {
        this.model = model;
    }

    /**
     * Returns the model's endpoint URL.
     *
     * @return the model's endpoint URL
     */
    public String getUrl() {
        return getUrl(false, false, null, null, null);
    }

    String getUrl(String baseUrl) {
        return getUrl(baseUrl, false);
    }

    String getUrl(String baseUrl, boolean stream) {
        return getUrl(stream, true, null, null, baseUrl);
    }

    /**
     * Returns the model's endpoint URL on Vertex.
     *
     * @return the model's endpoint URL on Vertex
     */
    public String getUrl(boolean stream) {
        return getUrl(stream, true, null, null, null);
    }

    /**
     * Returns the model's endpoint URL on Vertex with specified project and location.
     *
     * @return the model's endpoint URL on Vertex with specified project and location
     */
    String getUrl(boolean stream, String project, String location) {
        return getUrl(stream, true, project, location, null);
    }

    String getUrl(
            boolean stream, boolean useVertex, String project, String location, String baseUrl) {
        if (!useVertex) {
            if (baseUrl == null) {
                return "https://api.anthropic.com/v1/messages";
            }
            return baseUrl;
        }

        if (location == null) {
            location = Utils.getEnvOrSystemProperty("REGION", "global");
        }
        if (project == null) {
            project = Utils.getEnvOrSystemProperty("PROJECT");
        }
        if (baseUrl == null) {
            if ("global".equals(location)) {
                baseUrl = "https://aiplatform.googleapis.com";
            } else {
                baseUrl = "https://" + location + "-aiplatform.googleapis.com";
            }
        }
        StringBuilder sb = new StringBuilder(baseUrl);
        if (project == null) {
            throw new IllegalArgumentException("project is required.");
        }
        sb.append("/v1/projects/")
                .append(project)
                .append("/locations/")
                .append(location)
                .append("/publishers/anthropic/models/")
                .append(model)
                .append(':');
        if (stream) {
            sb.append("streamRawPredict");
        } else {
            sb.append("rawPredict");
        }
        return sb.toString();
    }
}
