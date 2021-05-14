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
import ai.djl.repository.License;
import java.beans.BeanInfo;
import java.beans.IntrospectionException;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.net.URI;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Transferring model-artifact-info.
 *
 * <p>this class inherits from ModelReferenceDTO and can so be used as a ModelReference.
 *
 * @author erik.bamberg@web.de
 */
public class ModelDTO extends ModelReferenceDTO {

    private static final Logger logger = LoggerFactory.getLogger(ModelDTO.class);

    private boolean snapshot;
    private String description;
    private String website;
    private URI repositoryURI;
    private Date lastUpdated;
    private String resourceType;
    private Map<String, String> licenses;

    private List<FileResourceDTO> files;

    private String dataset;
    private String layers;
    private String backbone;

    // a map of dynamic properties not mapped to any other field.
    private Map<String, Object> properties;

    /**
     * construct using reference data.
     *
     * @param name of model.
     * @param groupId of model.
     * @param artifactId of model.
     * @param version of model.
     */
    public ModelDTO(String name, String groupId, String artifactId, String version) {
        super(name, groupId, artifactId, version);
    }

    /**
     * constructs a model-dto from an artifact.
     *
     * @param artifact to receive information from.
     */
    public ModelDTO(Artifact artifact) {
        super(artifact);
        this.snapshot = artifact.isSnapshot();
        this.description = artifact.getMetadata().getDescription();
        this.website = artifact.getMetadata().getWebsite();
        this.repositoryURI = artifact.getResourceUri();
        this.lastUpdated = artifact.getMetadata().getLastUpdated();
        this.properties = new HashMap<>();

        licenses =
                Collections.unmodifiableMap(
                        artifact.getMetadata()
                                .getLicenses()
                                .values()
                                .stream()
                                .collect(Collectors.toMap(License::getName, License::getUrl)));

        this.resourceType = artifact.getMetadata().getResourceType();

        if (artifact.getProperties() != null) {
            this.dataset = artifact.getProperties().getOrDefault("dataset", "N/A");
            this.layers = artifact.getProperties().getOrDefault("layers", "N/A");
            this.backbone = artifact.getProperties().getOrDefault("backbone", "N/A");
            artifact.getProperties().forEach((k, v) -> properties.put(k, v));
        }

        // arguments
        Map<String, Object> arguments = artifact.getArguments(null);
        if (arguments != null) {
            try {
                arguments.forEach((k, v) -> properties.put(k, v));
            } catch (ClassCastException ex) {
                logger.error("Argument in model is not of expected type.", ex);
            }
        }

        Map<String, String> options = artifact.getOptions(null);
        if (options != null) {
            options.forEach((k, v) -> properties.put(k, v));
        }

        if (artifact.getFiles() != null) {
            files =
                    artifact.getFiles()
                            .entrySet()
                            .stream()
                            .map(e -> new FileResourceDTO(e.getKey(), e.getValue()))
                            .collect(Collectors.toList());
        } else {
            files = Collections.emptyList();
        }

        cleanUpDuplicateProperties();
    }

    /**
     * access the snapshot-property.
     *
     * @return the snapshot of this class.
     */
    public boolean isSnapshot() {
        return snapshot;
    }

    /**
     * access the description-property.
     *
     * @return the description of this Object.
     */
    public String getDescription() {
        return description;
    }

    /**
     * access the website-property.
     *
     * @return the website of this Object.
     */
    public String getWebsite() {
        return website;
    }

    /**
     * access the repositoryURI-property.
     *
     * @return the repositoryURI of this Object.
     */
    public URI getRepositoryURI() {
        return repositoryURI;
    }

    /**
     * access the lastUpdated-property.
     *
     * @return the lastUpdated of this Object.
     */
    public Date getLastUpdated() {
        return lastUpdated;
    }

    /**
     * access the resourceType-property.
     *
     * @return the resourceType of this Object.
     */
    public String getResourceType() {
        return resourceType;
    }

    /**
     * access the licenses-property.
     *
     * @return the licenses of this Object.
     */
    public Map<String, String> getLicenses() {
        return licenses;
    }

    /**
     * access the files-property.
     *
     * @return the files of this Object.
     */
    public List<FileResourceDTO> getFiles() {
        return files;
    }

    /**
     * access the dataset-property.
     *
     * @return the dataset of this Object.
     */
    public String getDataset() {
        return dataset;
    }

    /**
     * access the layers-property.
     *
     * @return the layers of this Object.
     */
    public String getLayers() {
        return layers;
    }

    /**
     * access the backbone-property.
     *
     * @return the backbone of this Object.
     */
    public String getBackbone() {
        return backbone;
    }

    /**
     * access the properties which is a list of model specific key/values.
     *
     * @return the properties of this object.
     */
    public Map<String, Object> getProperties() {
        return Collections.unmodifiableMap(properties);
    }

    protected final void cleanUpDuplicateProperties() {
        try {
            BeanInfo beanInfo = Introspector.getBeanInfo(this.getClass());
            Arrays.stream(beanInfo.getPropertyDescriptors())
                    .map(PropertyDescriptor::getName)
                    .forEach(properties::remove);
        } catch (IntrospectionException e) {
            logger.warn("cannot cleanup duplicate property fields", e);
        }
    }
}
