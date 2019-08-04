/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package software.amazon.ai.repository;

import java.net.URI;
import java.util.LinkedHashMap;
import java.util.Map;

@SuppressWarnings("PMD.LooseCoupling")
public class Artifact {

    private transient String metadataVersion;
    private transient String groupId;
    private transient String artifactId;
    private String version;
    private boolean snapshot;
    private LinkedHashMap<String, String> properties;
    private Map<String, Item> files;

    private URI baseUri;
    private transient Version cache;

    public String getMetadataVersion() {
        return metadataVersion;
    }

    public void setMetadataVersion(String metadataVersion) {
        this.metadataVersion = metadataVersion;
    }

    public String getGroupId() {
        return groupId;
    }

    public void setGroupId(String groupId) {
        this.groupId = groupId;
    }

    public String getArtifactId() {
        return artifactId;
    }

    public void setArtifactId(String artifactId) {
        this.artifactId = artifactId;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public boolean isSnapshot() {
        return snapshot;
    }

    public void setSnapshot(boolean snapshot) {
        this.snapshot = snapshot;
    }

    public Map<String, String> getProperties() {
        return properties;
    }

    public void setProperties(LinkedHashMap<String, String> properties) {
        this.properties = properties;
    }

    public URI getBaseUri() {
        return baseUri;
    }

    public URI getResourceUri() {
        URI uri = baseUri;
        if (properties != null) {
            for (String key : properties.keySet()) {
                uri = uri.resolve(key);
            }
        }
        return uri.resolve(version);
    }

    public void setBaseUri(URI baseUri) {
        this.baseUri = baseUri;
    }

    public Map<String, Item> getFiles() {
        files.forEach((k, v) -> v.setArtifact(this));
        return files;
    }

    public void setFiles(Map<String, Item> files) {
        this.files = files;
    }

    public boolean hasProperties(Map<String, String> filter) {
        if (filter == null || filter.isEmpty()) {
            return true;
        }

        if (properties == null || properties.isEmpty()) {
            return false;
        }

        for (Map.Entry<String, String> entry : filter.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            if (!value.equals(properties.get(key))) {
                return false;
            }
        }
        return true;
    }

    public Version getParsedVersion() {
        if (cache == null) {
            cache = new Version(version);
        }
        return cache;
    }

    public static final class Item {

        private String uri;
        private String sha1Hash;
        private Artifact artifact;

        public String getUri() {
            return uri;
        }

        public void setUri(String uri) {
            this.uri = uri;
        }

        public String getSha1Hash() {
            return sha1Hash;
        }

        public void setSha1Hash(String sha1Hash) {
            this.sha1Hash = sha1Hash;
        }

        public Artifact getArtifact() {
            return artifact;
        }

        public void setArtifact(Artifact artifact) {
            this.artifact = artifact;
        }
    }
}
