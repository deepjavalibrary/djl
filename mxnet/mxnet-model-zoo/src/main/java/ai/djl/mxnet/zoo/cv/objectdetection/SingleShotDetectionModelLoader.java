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
package ai.djl.mxnet.zoo.cv.objectdetection;

import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.SingleShotDetectionTranslator;
import ai.djl.modality.cv.transform.Resize;
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
import java.util.Map;

/**
 * Model loader for Single Shot Detection (SSD) models.
 *
 * <p>The model was trained on Gluon and loaded in DJL in MXNet Symbol Block. See <a
 * href="https://arxiv.org/pdf/1512.02325.pdf">SSD</a>.
 *
 * @see ai.djl.mxnet.engine.MxSymbolBlock
 */
public class SingleShotDetectionModelLoader
        extends BaseModelLoader<BufferedImage, DetectedObjects> {

    private static final Anchor BASE_ANCHOR = CV.OBJECT_DETECTION;
    private static final String GROUP_ID = MxModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "ssd";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public SingleShotDetectionModelLoader(Repository repository) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, ARTIFACT_ID), VERSION);
    }

    /** {@inheritDoc} */
    @Override
    public Translator<BufferedImage, DetectedObjects> getTranslator(Artifact artifact) {
        Map<String, Object> arguments = artifact.getArguments();
        int width = ((Double) arguments.getOrDefault("width", 512d)).intValue();
        int height = ((Double) arguments.getOrDefault("height", 512d)).intValue();
        double threshold = ((Double) arguments.getOrDefault("threshold", 0.2d));

        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(width, height)).add(new ToTensor());

        return new SingleShotDetectionTranslator.Builder()
                .setPipeline(pipeline)
                .setSynsetArtifactName("classes.txt")
                .optThreshold((float) threshold)
                .optRescaleSize(width, height)
                .build();
    }
}
