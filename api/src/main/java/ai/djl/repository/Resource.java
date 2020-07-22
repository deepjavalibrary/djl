/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.util.Progress;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class represents a resource in a {@link Repository}. */
public class Resource {

    private static final Logger logger = LoggerFactory.getLogger(Resource.class);

    private Repository repository;
    private MRL mrl;
    private String version;
    private Metadata metadata;

    /**
     * Constructs a {@code Resource} instance.
     *
     * @param repository the {@link Repository}
     * @param mrl the resource locator
     * @param version the version of the resource
     */
    public Resource(Repository repository, MRL mrl, String version) {
        this.repository = repository;
        this.mrl = mrl;
        this.version = version;
    }

    /**
     * Returns the {@link Repository} of the resource.
     *
     * @return the {@link Repository} of the resource
     */
    public Repository getRepository() {
        return repository;
    }

    /**
     * Returns the {@link MRL} of the resource.
     *
     * @return the {@link MRL} of the resource
     */
    public MRL getMrl() {
        return mrl;
    }

    /**
     * Returns the version of the resource.
     *
     * @return the version of the resource
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
        return repository.resolve(mrl, version, null);
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
            metadata = repository.locate(mrl);
            if (metadata == null) {
                throw new IOException("MRL: " + mrl + " resource not found.");
            }
        }
        return metadata;
    }
}
