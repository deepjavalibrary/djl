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
package ai.djl.genai.openai;

/** A data class represents chat completion schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Tool {

    private String type;
    private Function function;

    Tool(Builder builder) {
        type = builder.type;
        function = builder.function;
    }

    public String getType() {
        return type;
    }

    public Function getFunction() {
        return function;
    }

    /**
     * Creates a new function {@code Tool}.
     *
     * @param function the function for the tool
     * @return a new {@code Tool} instance
     */
    public static Tool of(Function function) {
        return builder().type("function").function(function).build();
    }

    /**
     * Creates a new function {@code Tool}.
     *
     * @param function the function for the tool
     * @return a new {@code Tool} instance
     */
    public static Tool of(Function.Builder function) {
        return builder().type("function").function(function).build();
    }

    /**
     * Creates a builder to build a {@code Tool}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code Tool}. */
    public static final class Builder {

        String type;
        Function function;

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder function(Function function) {
            this.function = function;
            return this;
        }

        public Builder function(Function.Builder function) {
            return function(function.build());
        }

        public Tool build() {
            return new Tool(this);
        }
    }
}
