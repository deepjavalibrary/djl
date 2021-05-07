/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.central.model.dto;

import ai.djl.repository.Artifact;
import java.net.URI;

/**
 * represents a file resource of a model.
 *
 * @author erik.bamberg@web.de
 */
public class FileResourceDTO {

    // TODO: Use the artifact repository to create base URI
    private static URI base = URI.create("https://mlrepo.djl.ai/");

    private String key;
    private String name;
    private String uri;
    private String sha1Hash;
    private String extension;
    private URI downloadLink;
    private long size;

    /**
     * constructs a FileResource object using the particular values.
     *
     * @param key of this file resource.
     * @param name of this file resource.
     * @param uri of this file resource.
     * @param sha1Hash of this file resource.
     * @param size of this file resource.
     */
    public FileResourceDTO(String key, String name, String uri, String sha1Hash, long size) {
        super();
        this.key = key;
        this.name = name;
        this.uri = uri;
        this.sha1Hash = sha1Hash;
        this.size = size;
    }

    /**
     * constructs a file resource from this item.
     *
     * @param key as an identifier.
     * @param item of an artifact.
     */
    public FileResourceDTO(String key, Artifact.Item item) {
        this.key = key;
        this.name = item.getName();
        this.uri = item.getUri();
        this.sha1Hash = item.getSha1Hash();
        this.size = item.getSize();
        this.extension = item.getExtension();

        URI fileUri = URI.create(item.getUri());
        URI baseUri = item.getArtifact().getMetadata().getRepositoryUri();
        if (!fileUri.isAbsolute()) {
            fileUri = base.resolve(baseUri).resolve(fileUri);
        }
        downloadLink = fileUri;
    }

    /**
     * access the key-property.
     *
     * @return the key of this class.
     */
    public String getKey() {
        return key;
    }
    /**
     * access the name-property.
     *
     * @return the name of this class.
     */
    public String getName() {
        return name;
    }
    /**
     * access the uri-property.
     *
     * @return the uri of this class.
     */
    public String getUri() {
        return uri;
    }
    /**
     * access the sha1Hash-property.
     *
     * @return the sha1Hash of this class.
     */
    public String getSha1Hash() {
        return sha1Hash;
    }
    /**
     * access the size-property.
     *
     * @return the size of this class.
     */
    public long getSize() {
        return size;
    }

    /**
     * access the extension-property.
     *
     * @return the extension of this class.
     */
    public String getExtension() {
        return extension;
    }

    /**
     * access the downloadLink-property.
     *
     * @return the downloadLink of this class.
     */
    public URI getDownloadLink() {
        return downloadLink;
    }
}
