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
package org.apache.mxnet.zoo.cv.segmentation;

import java.awt.image.BufferedImage;
import org.apache.mxnet.zoo.BaseModelLoader;
import org.apache.mxnet.zoo.ModelZoo;
import software.amazon.ai.modality.cv.DetectedObjects;
import software.amazon.ai.repository.Anchor;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.MRL.Model.CV;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.translate.Translator;

public class InstanceSegmentationModelLoader
        extends BaseModelLoader<BufferedImage, DetectedObjects> {

    private static final Anchor BASE_ANCHOR = CV.INSTANCE_SEGMENTATION;
    private static final String GROUP_ID = ModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "mask_rcnn";
    private static final String VERSION = "0.0.1";

    public InstanceSegmentationModelLoader(Repository repository) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, ARTIFACT_ID), VERSION);
    }

    @Override
    public Translator<BufferedImage, DetectedObjects> getTranslator() {
        return new InstanceSegementationTranslator.Builder()
                .setSynsetArtifactName("classes.txt")
                .optNormalize(
                        new float[] {0.485f, 0.456f, 0.406f}, new float[] {0.229f, 0.224f, 0.225f})
                .build();
    }
}
