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

import java.util.ArrayList;
import java.util.List;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Tool {

    private List<FunctionDeclaration> functionDeclarations;
    private GoogleSearch googleSearch;
    private GoogleSearchRetrieval googleSearchRetrieval;
    private Retrieval retrieval;

    Tool(Builder builder) {
        functionDeclarations = builder.functionDeclarations;
        googleSearch = builder.googleSearch;
        googleSearchRetrieval = builder.googleSearchRetrieval;
        retrieval = builder.retrieval;
    }

    public List<FunctionDeclaration> getFunctionDeclarations() {
        return functionDeclarations;
    }

    public GoogleSearch getGoogleSearch() {
        return googleSearch;
    }

    public GoogleSearchRetrieval getGoogleSearchRetrieval() {
        return googleSearchRetrieval;
    }

    public Retrieval getRetrieval() {
        return retrieval;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code Tool}. */
    public static final class Builder {

        List<FunctionDeclaration> functionDeclarations = new ArrayList<>();
        GoogleSearch googleSearch;
        GoogleSearchRetrieval googleSearchRetrieval;
        Retrieval retrieval;

        public Builder functionDeclarations(List<FunctionDeclaration> functionDeclarations) {
            this.functionDeclarations.clear();
            this.functionDeclarations.addAll(functionDeclarations);
            return this;
        }

        public Builder addFunctionDeclaration(FunctionDeclaration functionDeclaration) {
            this.functionDeclarations.add(functionDeclaration);
            return this;
        }

        public Builder addFunctionDeclaration(FunctionDeclaration.Builder functionDeclaration) {
            this.functionDeclarations.add(functionDeclaration.build());
            return this;
        }

        public Builder googleSearch(GoogleSearch googleSearch) {
            this.googleSearch = googleSearch;
            return this;
        }

        public Builder googleSearch(GoogleSearch.Builder googleSearch) {
            this.googleSearch = googleSearch.build();
            return this;
        }

        public Builder googleSearchRetrieval(GoogleSearchRetrieval googleSearchRetrieval) {
            this.googleSearchRetrieval = googleSearchRetrieval;
            return this;
        }

        public Builder googleSearchRetrieval(GoogleSearchRetrieval.Builder googleSearchRetrieval) {
            this.googleSearchRetrieval = googleSearchRetrieval.build();
            return this;
        }

        public Builder retrieval(Retrieval retrieval) {
            this.retrieval = retrieval;
            return this;
        }

        public Builder retrieval(Retrieval.Builder retrieval) {
            this.retrieval = retrieval.build();
            return this;
        }

        public Tool build() {
            return new Tool(this);
        }
    }
}
