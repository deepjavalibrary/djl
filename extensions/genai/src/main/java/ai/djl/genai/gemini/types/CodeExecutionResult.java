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
public class CodeExecutionResult {

    private Outcome outcome;
    private String output;

    CodeExecutionResult(Builder builder) {
        outcome = builder.outcome;
        output = builder.output;
    }

    public Outcome getOutcome() {
        return outcome;
    }

    public String getOutput() {
        return output;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code CodeExecutionResult}. */
    public static final class Builder {

        Outcome outcome;
        String output;

        public Builder outcome(Outcome outcome) {
            this.outcome = outcome;
            return this;
        }

        public Builder output(String output) {
            this.output = output;
            return this;
        }

        public CodeExecutionResult build() {
            return new CodeExecutionResult(this);
        }
    }
}
