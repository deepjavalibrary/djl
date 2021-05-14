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
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A ModelDTO with structure data-structures for computer vision models.
 *
 * @author erik.bamberg@web.de
 */
public class ComputerVisionModelDTO extends ModelDTO {

    private static final Logger logger = LoggerFactory.getLogger(ComputerVisionModelDTO.class);

    private double width;
    private double height;
    private boolean resize;
    private boolean rescale;

    /**
     * creates a ModelDTO for computer vision tasks.
     *
     * @param name of model.
     * @param groupId of model.
     * @param artifactId of model.
     * @param version of model.
     */
    public ComputerVisionModelDTO(String name, String groupId, String artifactId, String version) {
        super(name, groupId, artifactId, version);
    }

    /**
     * creates a ModelDTO for computer vision tasks.
     *
     * @param artifact to build this data transfer object.
     */
    public ComputerVisionModelDTO(Artifact artifact) {
        super(artifact);

        // arguments
        Map<String, Object> arguments = artifact.getArguments(null);
        if (arguments != null) {
            try {
                this.width = (double) arguments.getOrDefault("width", 0.0d);
                this.height = (double) arguments.getOrDefault("height", 0.0d);
                this.resize = (boolean) arguments.getOrDefault("resize", false);
                this.rescale = (boolean) arguments.getOrDefault("rescale", false);
            } catch (ClassCastException ex) {
                logger.error("Argument in model is not of expected type.", ex);
            }
        }
        cleanUpDuplicateProperties();
    }

    /**
     * access the width-property.
     *
     * @return the width of this class.
     */
    public double getWidth() {
        return width;
    }

    /**
     * access the height-property.
     *
     * @return the height of this class.
     */
    public double getHeight() {
        return height;
    }

    /**
     * access the resize-property.
     *
     * @return the resize of this class.
     */
    public boolean isResize() {
        return resize;
    }

    /**
     * access the rescale-property.
     *
     * @return the rescale of this class.
     */
    public boolean isRescale() {
        return rescale;
    }
}
