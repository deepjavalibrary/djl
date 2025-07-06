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

import java.util.Base64;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Part {

    private CodeExecutionResult codeExecutionResult;
    private ExecutableCode executableCode;
    private FileData fileData;
    private FunctionCall functionCall;
    private FunctionResponse functionResponse;
    private Blob inlineData;
    private String text;
    private Boolean thought;
    private String thoughtSignature;
    private VideoMetadata videoMetadata;

    Part(Builder builder) {
        codeExecutionResult = builder.codeExecutionResult;
        executableCode = builder.executableCode;
        fileData = builder.fileData;
        functionCall = builder.functionCall;
        functionResponse = builder.functionResponse;
        inlineData = builder.inlineData;
        text = builder.text;
        thought = builder.thought;
        thoughtSignature = builder.thoughtSignature;
        videoMetadata = builder.videoMetadata;
    }

    public CodeExecutionResult getCodeExecutionResult() {
        return codeExecutionResult;
    }

    public ExecutableCode getExecutableCode() {
        return executableCode;
    }

    public FileData getFileData() {
        return fileData;
    }

    public FunctionCall getFunctionCall() {
        return functionCall;
    }

    public FunctionResponse getFunctionResponse() {
        return functionResponse;
    }

    public Blob getInlineData() {
        return inlineData;
    }

    public String getText() {
        return text;
    }

    public Boolean getThought() {
        return thought;
    }

    public String getThoughtSignature() {
        return thoughtSignature;
    }

    public VideoMetadata getVideoMetadata() {
        return videoMetadata;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static Builder text(String text) {
        return builder().text(text);
    }

    public static Builder fileData(String fileUri, String mimeType) {
        return builder().fileData(FileData.builder().fileUri(fileUri).mimeType(mimeType));
    }

    public static Builder inlineData(byte[] bytes, String mimeType) {
        return builder().inlineData(Blob.builder().data(bytes).mimeType(mimeType));
    }

    /** Builder class for {@code Part}. */
    public static final class Builder {

        CodeExecutionResult codeExecutionResult;
        ExecutableCode executableCode;
        FileData fileData;
        FunctionCall functionCall;
        FunctionResponse functionResponse;
        Blob inlineData;
        String text;
        Boolean thought;
        String thoughtSignature;
        VideoMetadata videoMetadata;

        public Builder codeExecutionResult(CodeExecutionResult codeExecutionResult) {
            this.codeExecutionResult = codeExecutionResult;
            return this;
        }

        public Builder codeExecutionResult(CodeExecutionResult.Builder codeExecutionResult) {
            this.codeExecutionResult = codeExecutionResult.build();
            return this;
        }

        public Builder executableCode(ExecutableCode executableCode) {
            this.executableCode = executableCode;
            return this;
        }

        public Builder executableCode(ExecutableCode.Builder executableCode) {
            this.executableCode = executableCode.build();
            return this;
        }

        public Builder fileData(FileData fileData) {
            this.fileData = fileData;
            return this;
        }

        public Builder fileData(FileData.Builder fileData) {
            this.fileData = fileData.build();
            return this;
        }

        public Builder functionCall(FunctionCall functionCall) {
            this.functionCall = functionCall;
            return this;
        }

        public Builder functionCall(FunctionCall.Builder functionCall) {
            this.functionCall = functionCall.build();
            return this;
        }

        public Builder functionResponse(FunctionResponse functionResponse) {
            this.functionResponse = functionResponse;
            return this;
        }

        public Builder functionResponse(FunctionResponse.Builder functionResponse) {
            this.functionResponse = functionResponse.build();
            return this;
        }

        public Builder inlineData(Blob inlineData) {
            this.inlineData = inlineData;
            return this;
        }

        public Builder inlineData(Blob.Builder inlineData) {
            this.inlineData = inlineData.build();
            return this;
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Builder thought(Boolean thought) {
            this.thought = thought;
            return this;
        }

        public Builder thoughtSignature(byte[] thoughtSignature) {
            this.thoughtSignature = Base64.getEncoder().encodeToString(thoughtSignature);
            return this;
        }

        public Builder videoMetadata(VideoMetadata videoMetadata) {
            this.videoMetadata = videoMetadata;
            return this;
        }

        public Builder videoMetadata(VideoMetadata.Builder videoMetadata) {
            this.videoMetadata = videoMetadata.build();
            return this;
        }

        public Part build() {
            return new Part(this);
        }
    }
}
