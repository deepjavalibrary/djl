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

import java.net.URI;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Utility class to hold the Gemini models url. */
public final class Gemini {

    public static final Gemini GEMINI_2_5_PRO = builder().model("gemini-2.5-pro").build();
    public static final Gemini GEMINI_2_5_FLASH = builder().model("gemini-2.5-flash").build();
    public static final Gemini GEMINI_2_0_FLASH = builder().model("gemini-2.0-flash").build();
    public static final Gemini GEMINI_2_0_FLASH_LITE =
            builder().model("gemini-2.0-flash-lite").build();

    private static final Pattern PATTERN =
            Pattern.compile(
                    "/([^/]+)/(openai/chat/completions|(projects/([^/]+)/locations/([^/]+)/publishers/google/)?models/([^/]+):(.+))");

    private String url;

    Gemini(String url) {
        this.url = url;
    }

    /**
     * Returns the Gemini model URL.
     *
     * @return the Gemini model URL
     */
    public String getUrl() {
        return url;
    }

    /**
     * Returns the {@code Builder} based on current model.
     *
     * @return the {@code Builder} based on current model
     */
    public Builder toBuilder() {
        URI uri = URI.create(url);
        String baseUrl = uri.getScheme() + "://" + uri.getAuthority();
        String path = uri.getPath();
        Matcher m = PATTERN.matcher(path);
        if (!m.matches()) {
            throw new IllegalArgumentException("Invalid gemini URL: " + url);
        }
        String apiVersion = m.group(1);
        String project = m.group(4);
        String location = m.group(5);
        String model = m.group(6);
        String verb = m.group(7);
        boolean vertex = project != null;
        boolean chatCompletions = m.group(3) != null;

        return builder()
                .baseUrl(baseUrl)
                .model(model)
                .apiVersion(apiVersion)
                .useVertex(vertex)
                .project(project)
                .location(location)
                .verb(verb)
                .chatCompletions(chatCompletions);
    }

    /**
     * Creates a builder to build a {@code Gemini}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code Gemini}. */
    public static final class Builder {
        private String model;
        private String baseUrl;
        private boolean useVertex;
        private boolean chatCompletions;
        private String apiVersion;
        private String project = Utils.getenv("PROJECT");
        private String location = Utils.getenv("REGION", "global");
        private String verb = "generateContent";

        /**
         * Sets the model.
         *
         * @param model the model
         * @return the builder
         */
        public Builder model(String model) {
            this.model = model;
            return this;
        }

        /**
         * Sets the override base url.
         *
         * @param baseUrl the override base url
         * @return the builder
         */
        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        /**
         * Sets the api version.
         *
         * @param apiVersion the api version
         * @return the builder
         */
        public Builder apiVersion(String apiVersion) {
            this.apiVersion = apiVersion;
            return this;
        }

        /**
         * Sets if use Vertex endpoint.
         *
         * @param useVertex if use Vertex endpoint
         * @return the builder
         */
        public Builder useVertex(boolean useVertex) {
            this.useVertex = useVertex;
            return this;
        }

        /**
         * Sets the Vertex project.
         *
         * @param project the Vertex project
         * @return the builder
         */
        public Builder project(String project) {
            this.project = project;
            return this;
        }

        /**
         * Sets the Vertex location.
         *
         * @param location the Vertex location, default is global
         * @return the builder
         */
        public Builder location(String location) {
            this.location = location;
            return this;
        }

        /**
         * Sets the prediction verb.
         *
         * @param verb the prediction verb, default is generateContent
         * @return the builder
         */
        public Builder verb(String verb) {
            this.verb = verb;
            return this;
        }

        /**
         * Sets if use chat completions mode.
         *
         * @param chatCompletions if use chat completions mode
         * @return the builder
         */
        public Builder chatCompletions(boolean chatCompletions) {
            this.chatCompletions = chatCompletions;
            return this;
        }

        /**
         * Sets if use streamGenerateContent verb.
         *
         * @param stream if use streamGenerateContent verb
         * @return the builder
         */
        public Builder stream(boolean stream) {
            if (stream) {
                verb = "streamGenerateContent?alt=sse";
            }
            return this;
        }

        /**
         * Returns the {@code Gemini} instance.
         *
         * @return the {@code Gemini} instance
         */
        public Gemini build() {
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
                        .append(':')
                        .append(verb);
            } else {
                if (apiVersion == null) {
                    apiVersion = "v1beta";
                }
                sb.append(apiVersion).append("/models/").append(model).append(':').append(verb);
            }
            return new Gemini(sb.toString());
        }
    }
}
