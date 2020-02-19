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
import java.net.URI;

/**
 * The {@code MRL} (Machine learning Resource Locator) is a pointer to a {@link Metadata} "resource"
 * on a machine learning {@link Repository}.
 *
 * <p>Each mrl references a single metadata file (parsed to {@link Metadata} and the collection of
 * artifacts located within it. Those artifacts all share the same groupId and artifactId, but can
 * differ based on the name and properties.
 *
 * <p>The mrl consists of three different properties:
 *
 * <ul>
 *   <li>baseAnchor - The base anchor is used to organize metadata and artifacts into (multi-level)
 *       categories (See {@link Anchor}).
 *   <li>groupId - The group id identifies the group publishing the artifacts using a reverse domain
 *       name system.
 *   <li>artifactId - The artifact id identifies the different artifacts published by a single
 *       group.
 * </ul>
 */
public class MRL {

    private Anchor baseAnchor;
    private String groupId;
    private String artifactId;

    /**
     * Constructs an MRL.
     *
     * @param baseAnchor the desired anchor
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     */
    MRL(Anchor baseAnchor, String groupId, String artifactId) {
        this.baseAnchor = baseAnchor;
        this.groupId = groupId;
        this.artifactId = artifactId;
    }

    /**
     * Creates a model {@code MRL} with specified application.
     *
     * @param application the desired application
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @return a model {@code MRL}
     */
    public static MRL model(Application application, String groupId, String artifactId) {
        Anchor baseAnchor = Anchor.MODEL.resolve(application.getPath());
        return new MRL(baseAnchor, groupId, artifactId);
    }

    /**
     * Creates a dataset {@code MRL} with specified application.
     *
     * @param application the desired application
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @return a dataset {@code MRL}
     */
    public static MRL dataset(Application application, String groupId, String artifactId) {
        Anchor baseAnchor = Anchor.DATASET.resolve(application.getPath()).getParent();
        return new MRL(baseAnchor, groupId, artifactId);
    }

    /**
     * Returns the URI to the metadata location (used for {@link Repository} implementations).
     *
     * @return the URI to the metadata location
     */
    public URI toURI() {
        String groupIdPath = groupId.replace('.', '/');
        Anchor anchor = baseAnchor.resolve(groupIdPath, artifactId);
        return URI.create(anchor.getPath() + '/');
    }

    /**
     * Returns the base anchor.
     *
     * @return the base anchor
     */
    public Anchor getBaseAnchor() {
        return baseAnchor;
    }

    /**
     * Sets the base anchor.
     *
     * @param baseAnchor the new base anchor
     */
    public void setBaseAnchor(Anchor baseAnchor) {
        this.baseAnchor = baseAnchor;
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

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return toURI().toString();
    }
}
