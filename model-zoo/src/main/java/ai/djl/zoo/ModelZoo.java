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
package ai.djl.zoo;

import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.zoo.cv.classification.MlpModelLoader;
import ai.djl.zoo.cv.classification.ResNetModelLoader;
import ai.djl.zoo.cv.object_detection.ssd.SingleShotDetectionModelLoader;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/** Model Zoo is a repository that contains all models for DJL. */
public interface ModelZoo {

    String REPO_URL = "https://mlrepo.djl.ai/";
    Repository REPOSITORY = Repository.newInstance("zoo", REPO_URL);
    String GROUP_ID = "ai.djl.zoo";

    ResNetModelLoader RESNET = new ResNetModelLoader(REPOSITORY);
    MlpModelLoader MLP = new MlpModelLoader(REPOSITORY);
    SingleShotDetectionModelLoader SSD = new SingleShotDetectionModelLoader(REPOSITORY);

    /**
     * Gets the {@link ModelLoader} based on the model name.
     *
     * @param name the name of the model
     * @param <I> the input data type for preprocessing
     * @param <O> the output data type after postprocessing
     * @return the {@link ModelLoader} of the model
     * @throws ModelNotFoundException when the model cannot be found
     */
    @SuppressWarnings("unchecked")
    static <I, O> ModelLoader<I, O> getModelLoader(String name) throws ModelNotFoundException {
        try {
            Field field = ModelZoo.class.getDeclaredField(name);
            return (ModelLoader<I, O>) field.get(null);
        } catch (ReflectiveOperationException e) {
            throw new ModelNotFoundException(
                    "Model: " + name + " is not defined in MxModelZoo.", e);
        }
    }

    /**
     * Lists the available models in the ModelZoo.
     *
     * @return the list of all available models in loader format
     */
    static List<ModelLoader<?, ?>> listModels() {
        List<ModelLoader<?, ?>> list = new ArrayList<>();
        try {
            Field[] fields = ModelZoo.class.getDeclaredFields();
            for (Field field : fields) {
                if (field.getType().isAssignableFrom(ModelLoader.class)) {
                    list.add((ModelLoader<?, ?>) field.get(null));
                }
            }
        } catch (ReflectiveOperationException e) {
            // ignore
        }
        return list;
    }
}
