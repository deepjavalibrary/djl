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
public class FunctionCall {

    private Map<String, Object> args;
    private String id;
    private String name;

    FunctionCall(Builder builder) {
        args = builder.args;
        id = builder.id;
        name = builder.name;
    }

    public Map<String, Object> getArgs() {
        return args;
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code FunctionCall}. */
    public static final class Builder {

        Map<String, Object> args;
        String id;
        String name;

        public Builder args(Map<String, Object> args) {
            this.args = args;
            return this;
        }

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public FunctionCall build() {
            return new FunctionCall(this);
        }
    }
}
