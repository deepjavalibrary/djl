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
package ai.djl.mxnet.zoo.cv.segmentation;

import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.InstanceSegmentationTranslator;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.Anchor;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.MRL.Model.CV;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import java.awt.image.BufferedImage;

/**
 * Model loader for Instance Segmentation models.
 *
 * <p>The model was trained on Gluon and loaded in DJL in MXNet Symbol Block. See <a
 * href="https://arxiv.org/pdf/1703.06870.pdf">Mask R-CNN used in the model</a>.
 *
 * @see ai.djl.mxnet.engine.MxSymbolBlock
 */
public class InstanceSegmentationModelLoader
        extends BaseModelLoader<BufferedImage, DetectedObjects> {

    private static final Anchor BASE_ANCHOR = CV.INSTANCE_SEGMENTATION;
    private static final String GROUP_ID = MxModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "mask_rcnn";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public InstanceSegmentationModelLoader(Repository repository) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, ARTIFACT_ID), VERSION);
    }

    /** {@inheritDoc} */
    @Override
    public Translator<BufferedImage, DetectedObjects> getTranslator(Artifact artifact) {
        Pipeline pipeline = new Pipeline();
        pipeline.add(new ToTensor())
                .add(
                        new Normalize(
                                new float[] {0.485f, 0.456f, 0.406f},
                                new float[] {0.229f, 0.224f, 0.225f}));

        return new InstanceSegmentationTranslator.Builder()
                .setPipeline(pipeline)
                .setSynsetArtifactName("classes.txt")
                .build();
    }
}
