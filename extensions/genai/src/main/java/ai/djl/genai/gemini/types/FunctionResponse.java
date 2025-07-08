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
package ai.djl.genai.gemini.types;

import java.util.Map;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class FunctionResponse {

    private String id;
    private String name;
    private Map<String, Object> response;
    private FunctionResponseScheduling scheduling;
    private Boolean willContinue;

    FunctionResponse(Builder builder) {
        id = builder.id;
        name = builder.name;
        response = builder.response;
        scheduling = builder.scheduling;
        willContinue = builder.willContinue;
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public Map<String, Object> getResponse() {
        return response;
    }

    public FunctionResponseScheduling getScheduling() {
        return scheduling;
    }

    public Boolean getWillContinue() {
        return willContinue;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code FunctionResponse}. */
    public static final class Builder {

        String id;
        String name;
        Map<String, Object> response;
        FunctionResponseScheduling scheduling;
        Boolean willContinue;

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder response(Map<String, Object> response) {
            this.response = response;
            return this;
        }

        public Builder scheduling(FunctionResponseScheduling scheduling) {
            this.scheduling = scheduling;
            return this;
        }

        public Builder willContinue(Boolean willContinue) {
            this.willContinue = willContinue;
            return this;
        }

        public FunctionResponse build() {
            return new FunctionResponse(this);
        }
    }
}
