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

import ai.djl.Application;
import ai.djl.util.JsonUtils;
import java.io.Serializable;
import java.net.URI;
import java.util.Collections;
import java.util.Comparator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An {@code Artifact} is a set of data files such as a model or dataset.
 *
 * @see Repository
 */
@SuppressWarnings("PMD.LooseCoupling")
public class Artifact {

    private transient String metadataVersion;
    private String version;
    private boolean snapshot;
    private String name;
    private Map<String, String> properties;
    private Map<String, Object> arguments;
    private Map<String, String> options;
    private Map<String, Item> files;

    private transient Metadata metadata;
    private transient Version cache;

    /**
     * Returns the metadata format version.
     *
     * @return the metadata format version
     */
    public String getMetadataVersion() {
        return metadataVersion;
    }

    /**
     * Sets the metadata format version.
     *
     * @param metadataVersion the new version
     */
    public void setMetadataVersion(String metadataVersion) {
        this.metadataVersion = metadataVersion;
    }

    /**
     * Returns the artifact version.
     *
     * @return the artifact version
     * @see Version
     */
    public String getVersion() {
        return version;
    }

    /**
     * Sets the artifact version.
     *
     * @param version the new version
     * @see Version
     */
    public void setVersion(String version) {
        this.version = version;
    }

    /**
     * Returns true if the artifact is a snapshot.
     *
     * @return true if the artifact is a snapshot
     * @see Version
     */
    public boolean isSnapshot() {
        return snapshot;
    }

    /**
     * Sets if the artifact is a snapshot.
     *
     * @param snapshot true to make the artifact a snapshot
     * @see Version
     */
    public void setSnapshot(boolean snapshot) {
        this.snapshot = snapshot;
    }

    /**
     * Returns the artifact name.
     *
     * @return the artifact name
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the artifact name.
     *
     * @param name the new name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the artifact properties.
     *
     * @return the artifact properties
     * @see Repository
     */
    public Map<String, String> getProperties() {
        if (properties == null) {
            return Collections.emptyMap();
        }
        return properties;
    }

    /**
     * Sets the artifact properties.
     *
     * @param properties the new properties
     * @see Repository
     */
    public void setProperties(Map<String, String> properties) {
        this.properties = properties;
    }

    /**
     * Returns the artifact arguments.
     *
     * @param override the override configurations to the default arguments
     * @return the artifact arguments
     * @see Repository
     */
    public Map<String, Object> getArguments(Map<String, Object> override) {
        Map<String, Object> map = new ConcurrentHashMap<>();
        if (arguments != null) {
            map.putAll(arguments);
        }
        if (override != null) {
            map.putAll(override);
        }
        if (!map.containsKey("application") && metadata != null) {
            Application application = metadata.getApplication();
            if (application != null && Application.UNDEFINED != application) {
                map.put("application", application.getPath());
            }
        }
        return map;
    }

    /**
     * Returns the artifact arguments.
     *
     * @return the artifact arguments
     */
    public Map<String, Object> getArguments() {
        if (arguments == null) {
            arguments = new ConcurrentHashMap<>();
        }
        return arguments;
    }

    /**
     * Sets the artifact arguments.
     *
     * @param arguments the new arguments
     * @see Repository
     */
    public void setArguments(Map<String, Object> arguments) {
        this.arguments = arguments;
    }

    /**
     * Returns the artifact options.
     *
     * @param override the override options to the default options
     * @return the artifact options
     */
    public Map<String, String> getOptions(Map<String, String> override) {
        Map<String, String> map = new ConcurrentHashMap<>();
        if (options != null) {
            map.putAll(options);
        }
        if (override != null) {
            map.putAll(override);
        }
        return map;
    }

    /**
     * Sets the artifact arguments.
     *
     * @param options the new arguments
     */
    public void setOptions(Map<String, String> options) {
        this.options = options;
    }

    /**
     * Returns the metadata containing this artifact.
     *
     * @return the metadata containing this artifact
     * @see Repository
     */
    public Metadata getMetadata() {
        return metadata;
    }

    /**
     * Sets the associated metadata.
     *
     * @param metadata the new metadata
     * @see Repository
     */
    public void setMetadata(Metadata metadata) {
        this.metadata = metadata;
    }

    /**
     * Returns the location of the resource directory.
     *
     * @return the location of the resource directory
     */
    public URI getResourceUri() {
        URI uri = metadata.getRepositoryUri();
        if (properties != null) {
            for (String values : properties.values()) {
                uri = uri.resolve(values + '/');
            }
        }
        if (version == null) {
            return uri;
        }
        return uri.resolve(version + '/');
    }

    /**
     * Returns all the file items in the artifact.
     *
     * @return all the file items in the artifact
     */
    public Map<String, Item> getFiles() {
        if (files == null) {
            return Collections.emptyMap();
        }
        for (Map.Entry<String, Item> file : files.entrySet()) {
            file.getValue().setArtifact(this);
        }
        return files;
    }

    /**
     * Sets the file items.
     *
     * @param files the replacement file items
     */
    public void setFiles(Map<String, Item> files) {
        this.files = files;
    }

    /**
     * Returns true if every filter matches the corresponding property.
     *
     * @param filter the values to check against the properties
     * @return true if every filter matches the corresponding property
     * @see Repository
     */
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

    /**
     * Returns the artifact version as a {@link Version}.
     *
     * @return the artifact version as a {@link Version}
     * @see Version
     */
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
        if (metadata != null) {
            sb.append(metadata.getGroupId())
                    .append('/')
                    .append(metadata.getArtifactId())
                    .append('/');
        }
        if (version != null) {
            sb.append(version).append('/');
        }
        sb.append(name);
        if (properties != null) {
            sb.append(' ').append(JsonUtils.GSON.toJson(properties));
        } else {
            sb.append(" {}");
        }
        return sb.toString();
    }

    /** A file (possibly compressed) within an {@link Artifact}. */
    public static final class Item {

        private String uri;
        private String sha1Hash;
        private String name;
        private String type;
        private long size;
        private transient String extension;
        private transient Artifact artifact;

        /**
         * Returns the URI of the item.
         *
         * @return the URI of the item
         */
        public String getUri() {
            return uri;
        }

        /**
         * Sets the URI of the item.
         *
         * @param uri the new URI
         */
        public void setUri(String uri) {
            this.uri = uri;
        }

        /**
         * Returns the hash of the item.
         *
         * <p>This value is from the metadata, but should be checked when the item is downloaded.
         *
         * @return the sha1 hash
         */
        public String getSha1Hash() {
            return sha1Hash;
        }

        /**
         * Sets the sha1hash of the item.
         *
         * @param sha1Hash the new hash
         */
        public void setSha1Hash(String sha1Hash) {
            this.sha1Hash = sha1Hash;
        }

        /**
         * Sets the type of the item.
         *
         * <p>The valid types are:
         *
         * <ul>
         *   <li>"file" - used for single files and gzip compressed files
         *   <li>"dir" - used for extracted zip folders
         * </ul>
         *
         * @return the type string
         */
        public String getType() {
            if (type == null) {
                getExtension();
                if ("zip".equals(extension) || "tar".equals(extension) || "tgz".equals(extension)) {
                    type = "dir";
                } else {
                    type = "file";
                }
            }
            return type;
        }

        /**
         * Sets the type of the item.
         *
         * @param type the type
         * @see Item#getType()
         */
        public void setType(String type) {
            this.type = type;
        }

        /**
         * Returns the file size.
         *
         * @return the file size in bytes
         */
        public long getSize() {
            return size;
        }

        /**
         * Sets the file size.
         *
         * @param size the new size in bytes
         */
        public void setSize(long size) {
            this.size = size;
        }

        /**
         * Returns the item name.
         *
         * @return the item name
         */
        public String getName() {
            if (name == null) {
                int pos = uri.lastIndexOf('/');
                if (pos >= 0) {
                    name = uri.substring(pos + 1);
                } else {
                    name = uri;
                }
                name = FilenameUtils.getNamePart(name);
            }
            return name;
        }

        /**
         * Sets the item name.
         *
         * @param name the new name
         */
        public void setName(String name) {
            this.name = name;
        }

        /**
         * Returns the type of file extension.
         *
         * @return the type as "zip", "gzip", or "" for other
         */
        public String getExtension() {
            if (extension == null) {
                extension = FilenameUtils.getFileType(uri);
            }
            return extension;
        }

        /**
         * Sets the file extension.
         *
         * @param extension the new extension
         */
        public void setExtension(String extension) {
            this.extension = extension;
        }

        /**
         * Returns the artifact associated with this item.
         *
         * @return the artifact
         */
        public Artifact getArtifact() {
            return artifact;
        }

        /**
         * Sets the artifact associated with this item.
         *
         * @param artifact the new artifact
         */
        public void setArtifact(Artifact artifact) {
            this.artifact = artifact;
        }
    }

    /** A {@link Comparator} to compare artifacts based on their version numbers. */
    public static final class VersionComparator implements Comparator<Artifact>, Serializable {

        private static final long serialVersionUID = 1L;

        /** {@inheritDoc} */
        @Override
        public int compare(Artifact o1, Artifact o2) {
            return o1.getParsedVersion().compareTo(o2.getParsedVersion());
        }
    }
}
