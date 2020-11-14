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
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.modality.cv.zoo.ImageClassificationModelLoader;
import ai.djl.repository.Artifact;
import ai.djl.repository.Repository;
import java.util.List;
import java.util.Map;

/** Model loader for MLP models. */
public class MlpModelLoader extends ImageClassificationModelLoader {

    private static final String GROUP_ID = BasicModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "mlp";
    private static final String VERSION = "0.0.3";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public MlpModelLoader(Repository repository) {
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
        int width = ((Double) arguments.getOrDefault("width", 28d)).intValue();
        int height = ((Double) arguments.getOrDefault("height", 28d)).intValue();
        int input = width * height;
        int output = ((Double) arguments.get("output")).intValue();
        @SuppressWarnings("unchecked")
        int[] hidden =
                ((List<Double>) arguments.get("hidden"))
                        .stream()
                        .mapToInt(Double::intValue)
                        .toArray();

        Model model = Model.newInstance(name, device, engine);
        model.setBlock(new Mlp(input, output, hidden));
        return model;
    }
}
