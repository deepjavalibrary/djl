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

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class FunctionDeclaration {

    private Behavior behavior;
    private String description;
    private String name;
    private Schema parameters;
    private Object parametersJsonSchema;
    private Schema response;
    private Object responseJsonSchema;

    FunctionDeclaration(Builder builder) {
        behavior = builder.behavior;
        description = builder.description;
        name = builder.name;
        parameters = builder.parameters;
        parametersJsonSchema = builder.parametersJsonSchema;
        response = builder.response;
        responseJsonSchema = builder.responseJsonSchema;
    }

    public Behavior getBehavior() {
        return behavior;
    }

    public String getDescription() {
        return description;
    }

    public String getName() {
        return name;
    }

    public Schema getParameters() {
        return parameters;
    }

    public Object getParametersJsonSchema() {
        return parametersJsonSchema;
    }

    public Schema getResponse() {
        return response;
    }

    public Object getResponseJsonSchema() {
        return responseJsonSchema;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code FunctionDeclaration}. */
    public static final class Builder {

        Behavior behavior;
        String description;
        String name;
        Schema parameters;
        Object parametersJsonSchema;
        Schema response;
        Object responseJsonSchema;

        public Builder behavior(Behavior behavior) {
            this.behavior = behavior;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder parameters(Schema parameters) {
            this.parameters = parameters;
            return this;
        }

        public Builder parameters(Schema.Builder parameters) {
            this.parameters = parameters.build();
            return this;
        }

        public Builder parametersJsonSchema(Object parametersJsonSchema) {
            this.parametersJsonSchema = parametersJsonSchema;
            return this;
        }

        public Builder response(Schema response) {
            this.response = response;
            return this;
        }

        public Builder response(Schema.Builder response) {
            this.response = response.build();
            return this;
        }

        public Builder responseJsonSchema(Object responseJsonSchema) {
            this.responseJsonSchema = responseJsonSchema;
            return this;
        }

        public FunctionDeclaration build() {
            return new FunctionDeclaration(this);
        }
    }
}
