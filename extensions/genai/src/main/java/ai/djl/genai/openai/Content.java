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

import com.google.gson.annotations.SerializedName;

import java.util.Base64;

/** A data class represents chat completion schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Content {

    private String type;
    private String text;

    @SerializedName("image_url")
    private ImageContent imageUrl;

    private FileContent file;

    public Content(String text) {
        this.type = "text";
        this.text = text;
    }

    public Content(ImageContent imageUrl) {
        this.type = "image_url";
        this.imageUrl = imageUrl;
    }

    public Content(FileContent file) {
        this.type = "file";
        this.file = file;
    }

    public String getType() {
        return type;
    }

    public String getText() {
        return text;
    }

    public ImageContent getImageUrl() {
        return imageUrl;
    }

    public FileContent getFile() {
        return file;
    }

    public static Content fromText(String text) {
        return new Content(text);
    }

    public static Content fromImage(String imageUrl) {
        return new Content(new ImageContent(imageUrl));
    }

    public static Content fromFile(String id, byte[] data, String fileName) {
        String encoded = Base64.getEncoder().encodeToString(data);
        return new Content(new FileContent(id, encoded, fileName));
    }

    /** A data class represents chat completion schema. */
    public static final class ImageContent {

        private String url;

        public ImageContent(String url) {
            this.url = url;
        }

        public String getUrl() {
            return url;
        }
    }

    /** A data class represents chat completion schema. */
    public static final class FileContent {

        @SerializedName("file_data")
        private String fileData;

        @SerializedName("file_id")
        private String fileId;

        private String filename;

        FileContent(String fileData, String fileId, String filename) {
            this.fileData = fileData;
            this.fileId = fileId;
            this.filename = filename;
        }

        public String getFileData() {
            return fileData;
        }

        public String getFileId() {
            return fileId;
        }

        public String getFilename() {
            return filename;
        }
    }
}
