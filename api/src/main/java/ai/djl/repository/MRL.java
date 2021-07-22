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
import ai.djl.util.Progress;
import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    private static final Logger logger = LoggerFactory.getLogger(MRL.class);

    private String type;
    private Application application;
    private String groupId;
    private String artifactId;
    private String version;
    private String artifactName;
    private Repository repository;
    private Metadata metadata;

    /**
     * Constructs an MRL.
     *
     * @param repository the {@link Repository}
     * @param type the resource type
     * @param application the resource application
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @param version the resource version
     * @param artifactName the desired artifact name
     */
    private MRL(
            Repository repository,
            String type,
            Application application,
            String groupId,
            String artifactId,
            String version,
            String artifactName) {
        this.repository = repository;
        this.type = type;
        this.application = application;
        this.groupId = groupId;
        this.artifactId = artifactId;
        this.version = version;
        this.artifactName = artifactName;
    }

    /**
     * Creates a model {@code MRL} with specified application.
     *
     * @param repository the {@link Repository}
     * @param application the desired application
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @param version the resource version
     * @param artifactName the desired artifact name
     * @return a model {@code MRL}
     */
    public static MRL model(
            Repository repository,
            Application application,
            String groupId,
            String artifactId,
            String version,
            String artifactName) {
        return new MRL(
                repository, "model", application, groupId, artifactId, version, artifactName);
    }

    /**
     * Creates a dataset {@code MRL} with specified application.
     *
     * @param repository the {@link Repository}
     * @param application the desired application
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @param version the resource version
     * @return a dataset {@code MRL}
     */
    public static MRL dataset(
            Repository repository,
            Application application,
            String groupId,
            String artifactId,
            String version) {
        return new MRL(repository, "dataset", application, groupId, artifactId, version, null);
    }

    /**
     * Creates a dataset {@code MRL} with specified application.
     *
     * @param repository the {@link Repository}
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @return a dataset {@code MRL}
     */
    public static MRL undefined(Repository repository, String groupId, String artifactId) {
        return new MRL(repository, "", Application.UNDEFINED, groupId, artifactId, null, null);
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
     * Returns the repository.
     *
     * @return the repository
     */
    public Repository getRepository() {
        return repository;
    }

    /**
     * Returns the application.
     *
     * @return the application
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

    /**
     * Returns the version.
     *
     * @return the version
     */
    public String getVersion() {
        return version;
    }

    /**
     * Returns the default artifact.
     *
     * @return the default artifact
     * @throws IOException for various exceptions depending on the specific dataset
     */
    public Artifact getDefaultArtifact() throws IOException {
        return repository.resolve(this, null);
    }

    /**
     * Returns the first artifact that matches a given criteria.
     *
     * @param criteria the criteria to match against
     * @return the first artifact that matches the criteria. Null will be returned if no artifact
     *     matches
     * @throws IOException for errors while loading the model
     */
    public Artifact match(Map<String, String> criteria) throws IOException {
        List<Artifact> list = search(criteria);
        if (list.isEmpty()) {
            return null;
        }
        if (artifactName != null) {
            for (Artifact artifact : list) {
                if (artifactName.equals(artifact.getName())) {
                    return artifact;
                }
            }
            return null;
        }
        return list.get(0);
    }

    /**
     * Returns a list of artifacts in this resource.
     *
     * @return a list of artifacts in this resource
     * @throws IOException for errors while loading the model
     */
    public List<Artifact> listArtifacts() throws IOException {
        return getMetadata().getArtifacts();
    }

    /**
     * Prepares the artifact for use.
     *
     * @param artifact the artifact to prepare
     * @throws IOException if it failed to prepare
     */
    public void prepare(Artifact artifact) throws IOException {
        prepare(artifact, null);
    }

    /**
     * Prepares the artifact for use with progress tracking.
     *
     * @param artifact the artifact to prepare
     * @param progress the progress tracker
     * @throws IOException if it failed to prepare
     */
    public void prepare(Artifact artifact, Progress progress) throws IOException {
        if (artifact != null) {
            logger.debug("Preparing artifact: {}, {}", repository.getName(), artifact);
            repository.prepare(artifact, progress);
        }
    }

    /**
     * Returns all the artifacts that match a given criteria.
     *
     * @param criteria the criteria to match against
     * @return all the artifacts that match a given criteria
     * @throws IOException for errors while loading the model
     */
    private List<Artifact> search(Map<String, String> criteria) throws IOException {
        return getMetadata().search(VersionRange.parse(version), criteria);
    }

    private Metadata getMetadata() throws IOException {
        if (metadata == null) {
            metadata = repository.locate(this);
            if (metadata == null) {
                throw new IOException(this + " resource not found.");
            }
        }
        return metadata;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return toURI().toString();
    }
}
