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

import ai.djl.util.Utils;

/** Utility class to hold the Gemini models url. */
public final class Gemini {

    public static final Gemini GEMINI_2_5_PRO = new Gemini("gemini-2.5-pro");
    public static final Gemini GEMINI_2_5_FLASH = new Gemini("gemini-2.5-flash");
    public static final Gemini GEMINI_2_0_FLASH = new Gemini("gemini-2.0-flash");
    public static final Gemini GEMINI_2_0_FLASH_LITE = new Gemini("gemini-2.0-flash-lite");

    private String model;

    Gemini(String model) {
        this.model = model;
    }

    /**
     * Returns the model name.
     *
     * @return the model name.
     */
    public String name() {
        return model;
    }

    /**
     * Returns the chatcompletions compatible endpoint URL.
     *
     * @return the chatcompletions compatible endpoint URL
     */
    public String getChatCompletionsUrl() {
        return getUrl(false, true, false, null, null, null, null);
    }

    /**
     * Returns the chatcompletions compatible endpoint URL with override base URL.
     *
     * @return the chatcompletions compatible endpoint URL with override base URL
     */
    public String getChatCompletionsUrl(String baseUrl) {
        return getUrl(false, true, false, null, null, null, baseUrl);
    }

    /**
     * Returns the model's endpoint URL.
     *
     * @return the model's endpoint URL
     */
    public String getUrl() {
        return getUrl(null);
    }

    /**
     * Returns the model's streaming endpoint URL.
     *
     * @return the model's streaming endpoint URL
     */
    public String getUrl(boolean stream) {
        return getUrl(stream, false);
    }

    /**
     * Returns the model's streaming endpoint URL on Vertex.
     *
     * @return the model's streaming endpoint URL on Vertex
     */
    public String getUrl(boolean stream, boolean useVertex) {
        return getUrl(stream, false, useVertex, null, null, null, null);
    }

    /**
     * Returns the endpoint URL with override base URL.
     *
     * @return the endpoint URL with override base URL
     */
    public String getUrl(String baseUrl) {
        return getUrl(false, false, false, null, null, null, baseUrl);
    }

    /**
     * Returns the endpoint URL with override base URL.
     *
     * @return the endpoint URL with override base URL
     */
    public String getUrl(String baseUrl, boolean stream) {
        return getUrl(stream, false, true, null, null, null, baseUrl);
    }

    /**
     * Returns the model's endpoint URL with specified project and location.
     *
     * @return the model's endpoint URL with specified project and location
     */
    public String getUrl(boolean stream, String project, String location) {
        return getUrl(stream, project, location, null);
    }

    /**
     * Returns the vertex model's endpoint URL with specified project, location and api version.
     *
     * @return the vertex model's endpoint URL with specified project, location and api version
     */
    public String getUrl(boolean stream, String project, String location, String apiVersion) {
        return getUrl(stream, false, true, project, location, apiVersion, null);
    }

    String getUrl(
            boolean stream,
            boolean chatCompletions,
            boolean useVertex,
            String project,
            String location,
            String apiVersion,
            String baseUrl) {
        if (location == null) {
            location = Utils.getEnvOrSystemProperty("REGION", "global");
        }
        if (project == null) {
            project = Utils.getEnvOrSystemProperty("PROJECT");
        }
        if (baseUrl == null) {
            if (useVertex && !"global".equals(location)) {
                baseUrl = "https://" + location + "-generativelanguage.googleapis.com";
            } else {
                baseUrl = "https://generativelanguage.googleapis.com";
            }
        }
        StringBuilder sb = new StringBuilder(baseUrl);
        sb.append('/');
        if (chatCompletions) {
            if (apiVersion == null) {
                apiVersion = "v1beta";
            }
            sb.append(apiVersion).append("/openai/chat/completions");
            return sb.toString();
        } else if (useVertex) {
            if (project == null) {
                throw new IllegalArgumentException("project is required.");
            }
            if (apiVersion == null) {
                apiVersion = "v1";
            }
            sb.append(apiVersion)
                    .append("/projects/")
                    .append(project)
                    .append("/locations/")
                    .append(location)
                    .append("/publishers/google/models/")
                    .append(model)
                    .append(':');
        } else {
            if (apiVersion == null) {
                apiVersion = "v1beta";
            }
            sb.append(apiVersion).append("/models/").append(model).append(':');
        }
        if (stream) {
            sb.append("streamGenerateContent?alt=sse");
        } else {
            sb.append("generateContent");
        }
        return sb.toString();
    }
}
