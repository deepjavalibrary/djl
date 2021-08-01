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
import ai.djl.repository.zoo.DefaultModelZoo;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * A {@code Metadata} is a collection of {@link Artifact}s with unified metadata (including {@link
 * MRL}) that are stored in the same "metadata.json" file.
 *
 * <p>All of the artifacts located within the metadata share the data defined at the metadata level
 * such as name, description, and website. The key difference between the artifacts within the same
 * metadata are the properties.
 *
 * @see Repository
 */
public class Metadata {

    private String metadataVersion;
    private String resourceType;
    private String application;
    protected String groupId;
    protected String artifactId;
    private String name;
    private String description;
    private String website;
    protected Map<String, License> licenses;
    protected List<Artifact> artifacts;
    private Date lastUpdated;
    private transient Application applicationClass;
    private transient URI repositoryUri;

    /**
     * Returns the artifacts matching the version and property requirements.
     *
     * @param versionRange the version range for the artifact
     * @param filter the property filter
     * @return the matching artifacts
     */
    public List<Artifact> search(VersionRange versionRange, Map<String, String> filter) {
        List<Artifact> results = versionRange.matches(artifacts);
        if (filter == null) {
            return results;
        }

        return results.stream().filter(a -> a.hasProperties(filter)).collect(Collectors.toList());
    }

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
     * Returns the resource type.
     *
     * @return the resource type
     */
    public String getResourceType() {
        return resourceType;
    }

    /**
     * Returns the resource type.
     *
     * @param resourceType the resource type
     */
    public void setResourceType(String resourceType) {
        this.resourceType = resourceType;
    }

    /**
     * Returns the groupId.
     *
     * @return the groupId
     */
    public String getGroupId() {
        return groupId;
    }

    /**
     * Sets the groupId.
     *
     * @param groupId the new groupId
     */
    public void setGroupId(String groupId) {
        this.groupId = groupId;
    }

    /**
     * Returns the artifactId.
     *
     * @return the artifactId
     */
    public String getArtifactId() {
        return artifactId;
    }

    /**
     * Sets the artifactId.
     *
     * @param artifactId the new artifactId
     */
    public void setArtifactId(String artifactId) {
        this.artifactId = artifactId;
    }

    /**
     * Returns the metadata-level name.
     *
     * @return the metadata-level name
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the metadata-level name.
     *
     * @param name the new metadata-level name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the description.
     *
     * @return the description
     */
    public String getDescription() {
        return description;
    }

    /**
     * Sets the description.
     *
     * @param description the description
     */
    public void setDescription(String description) {
        this.description = description;
    }

    /**
     * Returns the website.
     *
     * @return the website
     */
    public String getWebsite() {
        return website;
    }

    /**
     * Sets the website.
     *
     * @param website the website
     */
    public void setWebsite(String website) {
        this.website = website;
    }

    /**
     * Returns the {@link Application}.
     *
     * @return the {@link Application}
     */
    public Application getApplication() {
        if (applicationClass == null && application != null) {
            applicationClass = Application.of(application);
        }
        return applicationClass;
    }

    /**
     * Sets the {@link Application}.
     *
     * @param application {@link Application}
     */
    public final void setApplication(Application application) {
        this.applicationClass = application;
        this.application = application.getPath();
    }

    /**
     * Returns the {@link License}.
     *
     * @return licenses in this metadata
     */
    public Map<String, License> getLicenses() {
        return licenses;
    }

    /**
     * Sets the {@link License}.
     *
     * @param licenses {@link License}
     */
    public void setLicense(Map<String, License> licenses) {
        this.licenses = licenses;
    }

    /**
     * Adds one {@link License}.
     *
     * @param license {@link License}
     */
    public void addLicense(License license) {
        if (licenses == null) {
            licenses = new ConcurrentHashMap<>();
        }
        licenses.put(license.getId(), license);
    }

    /**
     * Returns all the artifacts in the metadata.
     *
     * @return the artifacts in the metadata
     */
    public List<Artifact> getArtifacts() {
        return artifacts;
    }

    /**
     * Sets the artifacts for the metadata.
     *
     * @param artifacts the new artifacts
     */
    public void setArtifacts(List<Artifact> artifacts) {
        this.artifacts = artifacts;
        for (Artifact artifact : artifacts) {
            artifact.setMetadata(this);
        }
    }

    /**
     * Adds one artifact for the metadata.
     *
     * @param artifact the new artifact
     */
    public void addArtifact(Artifact artifact) {
        if (artifacts == null) {
            artifacts = new ArrayList<>();
        }
        artifacts.add(artifact);
    }

    /**
     * Returns the last update date for the metadata.
     *
     * @return the last update date
     */
    public Date getLastUpdated() {
        return lastUpdated;
    }

    /**
     * Sets the last update date for the metadata.
     *
     * @param lastUpdated the new last update date
     */
    public void setLastUpdated(Date lastUpdated) {
        this.lastUpdated = lastUpdated;
    }

    /**
     * Returns the URI to the repository storing the metadata.
     *
     * @return the URI to the repository storing the metadata
     */
    public URI getRepositoryUri() {
        return repositoryUri;
    }

    /**
     * Sets the repository URI.
     *
     * @param repositoryUri the new URI
     */
    public void setRepositoryUri(URI repositoryUri) {
        this.repositoryUri = repositoryUri;
    }

    /**
     * Restores artifacts state.
     *
     * <p>This call is required after the metadata is restored back from JSON.
     *
     * @param arguments the override arguments
     */
    public final void init(Map<String, String> arguments) {
        if (artifacts != null) {
            for (Artifact artifact : artifacts) {
                artifact.setMetadata(this);
                artifact.getArguments().putAll(arguments);
            }
        }
    }

    /** A {@code Metadata} class that matches all any search criteria. */
    public static final class MatchAllMetadata extends Metadata {

        /** Creates a {@code MatchAllMetadata} instance. */
        public MatchAllMetadata() {
            groupId = DefaultModelZoo.GROUP_ID;
            artifacts = Collections.emptyList();
            setApplication(Application.UNDEFINED);
        }

        /** {@inheritDoc} */
        @Override
        public List<Artifact> search(VersionRange versionRange, Map<String, String> filter) {
            return getArtifacts();
        }
    }
}
