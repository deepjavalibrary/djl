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
package ai.djl.basicmodelzoo.cv.classification;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1.Builder;
import ai.djl.modality.cv.zoo.ImageClassificationModelLoader;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.Artifact;
import ai.djl.repository.Repository;
import java.util.List;
import java.util.Map;

/** Model loader for ResNet_V1. */
public class ResNetModelLoader extends ImageClassificationModelLoader {

    private static final String GROUP_ID = BasicModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "resnet";
    private static final String VERSION = "0.0.2";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public ResNetModelLoader(Repository repository) {
        super(repository, GROUP_ID, ARTIFACT_ID, VERSION, new BasicModelZoo());
    }

    /** {@inheritDoc} */
    @Override
    protected Model createModel(
            String name,
            Device device,
            Artifact artifact,
            Map<String, Object> arguments,
            String engine) {
        Model model = Model.newInstance(name, device, engine);
        model.setBlock(resnetBlock(arguments));
        return model;
    }

    private Block resnetBlock(Map<String, Object> arguments) {
        @SuppressWarnings("unchecked")
        Shape shape =
                new Shape(
                        ((List<Double>) arguments.get("imageShape"))
                                .stream()
                                .mapToLong(Double::longValue)
                                .toArray());
        Builder blockBuilder =
                ResNetV1.builder()
                        .setNumLayers((int) ((double) arguments.get("numLayers")))
                        .setOutSize((long) ((double) arguments.get("outSize")))
                        .setImageShape(shape);
        if (arguments.containsKey("batchNormMomentum")) {
            float batchNormMomentum = (float) ((double) arguments.get("batchNormMomentum"));
            blockBuilder.optBatchNormMomentum(batchNormMomentum);
        }
        return blockBuilder.build();
    }
}
