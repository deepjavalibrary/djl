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
package ai.djl.repository;

import java.io.Serializable;
import java.net.URI;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;

@SuppressWarnings("PMD.LooseCoupling")
public class Artifact {
    private transient String metadataVersion;
    private transient String groupId;
    private transient String artifactId;
    private String version;
    private boolean snapshot;
    private String name;
    private LinkedHashMap<String, String> properties;
    private LinkedHashMap<String, Object> arguments;
    private Map<String, Item> files;

    private transient Metadata metadata;
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

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Map<String, String> getProperties() {
        if (properties == null) {
            return Collections.emptyMap();
        }
        return properties;
    }

    public void setProperties(LinkedHashMap<String, String> properties) {
        this.properties = properties;
    }

    public Map<String, Object> getArguments() {
        if (arguments == null) {
            return Collections.emptyMap();
        }
        return arguments;
    }

    public void setArguments(LinkedHashMap<String, Object> arguments) {
        this.arguments = arguments;
    }

    public Metadata getMetadata() {
        return metadata;
    }

    public void setMetadata(Metadata metadata) {
        this.metadata = metadata;
    }

    public URI getResourceUri() {
        URI uri = metadata.getRepositoryUri();
        if (properties != null) {
            for (String values : properties.values()) {
                uri = uri.resolve(values + '/');
            }
        }
        return uri.resolve(version + '/');
    }

    public Map<String, Item> getFiles() {
        if (files == null) {
            return Collections.emptyMap();
        }
        for (Map.Entry<String, Item> file : files.entrySet()) {
            file.getValue().setArtifact(this);
            if (file.getValue().name == null && "dir".equals(file.getValue().getType())) {
                file.getValue().name = file.getKey();
            }
        }
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

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(100);
        sb.append(name).append(':').append(version).append(" {");
        boolean first = true;
        for (Map.Entry<String, String> entry : properties.entrySet()) {
            if (first) {
                first = false;
            } else {
                sb.append(" ,");
            }
            sb.append('"')
                    .append(entry.getKey())
                    .append("\": \"")
                    .append(entry.getValue())
                    .append('"');
        }
        sb.append('}');
        return sb.toString();
    }

    public static final class Item {

        private String uri;
        private String sha1Hash;
        private String name;
        private String type;
        private long size;
        private String extension;
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

        public String getType() {
            if (type == null) {
                getExtension();
                if ("zip".equals(extension)) {
                    type = "dir";
                } else {
                    type = "file";
                }
            }
            return type;
        }

        public void setType(String type) {
            this.type = type;
        }

        public long getSize() {
            return size;
        }

        public void setSize(long size) {
            this.size = size;
        }

        public String getName() {
            if (name == null) {
                if ("dir".equals(getType())) {
                    name = "";
                } else {
                    int pos = uri.lastIndexOf('/');
                    if (pos >= 0) {
                        name = uri.substring(pos + 1);
                    } else {
                        name = uri;
                    }
                    if (name.endsWith(".z") || name.endsWith(".gz") || name.endsWith(".zip")) {
                        pos = name.lastIndexOf('.');
                        if (pos > 0) {
                            name = name.substring(0, pos);
                        }
                    }
                }
            }
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getExtension() {
            if (extension == null) {
                if (uri.endsWith(".zip")) {
                    extension = "zip";
                } else if (uri.endsWith(".gz") || uri.endsWith(".z")) {
                    extension = "gzip";
                } else {
                    extension = "";
                }
            }
            return extension;
        }

        public void setExtension(String extension) {
            this.extension = extension;
        }

        public Artifact getArtifact() {
            return artifact;
        }

        public void setArtifact(Artifact artifact) {
            this.artifact = artifact;
        }
    }

    public static final class VersionComparator implements Comparator<Artifact>, Serializable {

        private static final long serialVersionUID = 1L;

        /** {@inheritDoc} */
        @Override
        public int compare(Artifact o1, Artifact o2) {
            return o1.getParsedVersion().compareTo(o2.getParsedVersion());
        }
    }
}
