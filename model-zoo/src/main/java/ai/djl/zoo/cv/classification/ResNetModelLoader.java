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
package ai.djl.zoo.cv.classification;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.modality.Classification;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.Anchor;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.MRL.Model.CV;
import ai.djl.repository.Repository;
import ai.djl.translate.Translator;
import ai.djl.zoo.BaseModelLoader;
import ai.djl.zoo.ModelZoo;
import ai.djl.zoo.cv.classification.ResNetV1.Builder;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public class ResNetModelLoader extends BaseModelLoader<BufferedImage, Classification> {

    private static final Anchor BASE_ANCHOR = CV.IMAGE_CLASSIFICATION;
    private static final String GROUP_ID = ModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "resnetv1";
    private static final String VERSION = "0.0.1";

    public ResNetModelLoader(Repository repository) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, ARTIFACT_ID), VERSION);
    }

    @Override
    public Translator<BufferedImage, Classification> getTranslator() {
        return new ImageClassificationTranslator.Builder()
                .optCenterCrop()
                .optResize(224, 224)
                .setSynsetArtifactName("synset.txt")
                .build();
    }

    @Override
    @SuppressWarnings("unchecked")
    protected Model loadModel(Artifact artifact, Path modelPath, Device device) throws IOException {
        Model model = Model.newInstance(device);
        Map<String, Object> arguments = artifact.getArguments();
        Builder blockBuilder =
                new ResNetV1.Builder()
                        .setNumLayers((int) ((double) arguments.get("numLayers")))
                        .setOutSize((long) ((double) arguments.get("outSize")))
                        .setImageShape(
                                new Shape(
                                        ((List<Double>) arguments.get("imageShape"))
                                                .stream()
                                                .mapToLong(Double::longValue)
                                                .toArray()));
        if (arguments.containsKey("batchNormMomentum")) {
            blockBuilder.setBatchNormMomemtum((float) arguments.get("batchNormMomentum"));
        }
        Block block = blockBuilder.build();
        model.setBlock(block);
        model.load(repository.getFile(artifact.getFiles().get("parameters"), ""), "resnet");
        return model;
    }
}
