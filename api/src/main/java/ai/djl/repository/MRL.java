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
 *   <li>type - The resource type, e.g. model or dataset.
 *   <li>application - The resource application (See {@link Application}).
 *   <li>groupId - The group id identifies the group publishing the artifacts using a reverse domain
 *       name system.
 *   <li>artifactId - The artifact id identifies the different artifacts published by a single
 *       group.
 * </ul>
 */
public final class MRL {

    private String type;
    private Application application;
    private String groupId;
    private String artifactId;

    /**
     * Constructs an MRL.
     *
     * @param type the resource type
     * @param application the resource application
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     */
    private MRL(String type, Application application, String groupId, String artifactId) {
        this.type = type;
        this.application = application;
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
        return new MRL("model", application, groupId, artifactId);
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
        return new MRL("dataset", application, groupId, artifactId);
    }

    /**
     * Creates a dataset {@code MRL} with specified application.
     *
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @return a dataset {@code MRL}
     */
    public static MRL undefined(String groupId, String artifactId) {
        return new MRL("", Application.UNDEFINED, groupId, artifactId);
    }

    /**
     * Returns the URI to the metadata location (used for {@link Repository} implementations).
     *
     * @return the URI to the metadata location
     */
    public URI toURI() {
        StringBuilder sb = new StringBuilder();
        if (!type.isEmpty()) {
            sb.append(type).append('/');
        }
        sb.append(application.getPath())
                .append('/')
                .append(groupId.replace('.', '/'))
                .append('/')
                .append(artifactId)
                .append('/');

        return URI.create(sb.toString());
    }

    /**
     * Returns the resource application.
     *
     * @return the resource application
     */
    public Application getApplication() {
        return application;
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
     * Returns the artifactId.
     *
     * @return the artifactId
     */
    public String getArtifactId() {
        return artifactId;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return toURI().toString();
    }
}
