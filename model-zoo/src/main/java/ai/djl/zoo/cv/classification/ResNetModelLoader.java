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
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.ImageClassificationTranslator;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.Anchor;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.MRL.Model.CV;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.zoo.ModelZoo;
import ai.djl.zoo.cv.classification.ResNetV1.Builder;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** Model loader for ResNet_V1. */
public class ResNetModelLoader extends BaseModelLoader<BufferedImage, Classifications> {

    private static final Anchor BASE_ANCHOR = CV.IMAGE_CLASSIFICATION;
    private static final String GROUP_ID = ModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "resnet";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public ResNetModelLoader(Repository repository) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, ARTIFACT_ID), VERSION);
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public Translator<BufferedImage, Classifications> getTranslator(Artifact artifact) {
        Map<String, Object> arguments = artifact.getArguments();
        List<Double> shape = (List<Double>) arguments.get("imageShape");
        int width = shape.get(2).intValue();
        int height = shape.get(1).intValue();

        Pipeline pipeline = new Pipeline();
        pipeline.add(new CenterCrop()).add(new Resize(width, height)).add(new ToTensor());
        return new ImageClassificationTranslator.Builder()
                .setPipeline(pipeline)
                .setSynsetArtifactName("synset.txt")
                .build();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    protected Model loadModel(Artifact artifact, Path modelPath, Device device)
            throws IOException, MalformedModelException {
        Map<String, Object> arguments = artifact.getArguments();
        Shape shape =
                new Shape(
                        ((List<Double>) arguments.get("imageShape"))
                                .stream()
                                .mapToLong(Double::longValue)
                                .toArray());
        Builder blockBuilder =
                new ResNetV1.Builder()
                        .setNumLayers((int) ((double) arguments.get("numLayers")))
                        .setOutSize((long) ((double) arguments.get("outSize")))
                        .setImageShape(shape);
        if (arguments.containsKey("batchNormMomentum")) {
            float batchNormMomentum = (float) ((double) arguments.get("batchNormMomentum"));
            blockBuilder.optBatchNormMomemtum(batchNormMomentum);
        }
        Block block = blockBuilder.build();

        Model model = Model.newInstance(device);
        model.setBlock(block);
        model.load(modelPath, artifact.getName());
        return model;
    }
}
