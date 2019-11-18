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
package ai.djl.repository.zoo;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.repository.Artifact;
import ai.djl.util.Progress;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * A ModelLoader loads a particular {@link ZooModel} from a Repository for a model zoo.
 *
 * @param <I> the input type to the loaded model
 * @param <O> the output type of the loaded model
 */
public interface ModelLoader<I, O> {

    /**
     * Loads a model.
     *
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    default ZooModel<I, O> loadModel()
            throws IOException, ModelNotFoundException, MalformedModelException {
        return loadModel(null, null, null);
    }

    /**
     * Loads a model.
     *
     * @param progress the progress tracker to update while loading the model
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    default ZooModel<I, O> loadModel(Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        return loadModel(null, null, progress);
    }

    /**
     * Loads the model with the given criteria.
     *
     * @param criteria the criteria to match against the loaded model
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    default ZooModel<I, O> loadModel(Map<String, String> criteria)
            throws IOException, ModelNotFoundException, MalformedModelException {
        return loadModel(criteria, null, null);
    }

    /**
     * Loads the model with the given criteria.
     *
     * @param criteria the criteria to match against the loaded model
     * @param progress the progress tracker to update while loading the model
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    default ZooModel<I, O> loadModel(Map<String, String> criteria, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        return loadModel(criteria, null, progress);
    }

    /**
     * Loads the model with the given criteria.
     *
     * @param criteria the criteria to match against the loaded model
     * @param device the device the loaded model should use
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    default ZooModel<I, O> loadModel(Map<String, String> criteria, Device device)
            throws IOException, ModelNotFoundException, MalformedModelException {
        return loadModel(criteria, device, null);
    }

    /**
     * Loads the model with the given criteria.
     *
     * @param criteria the criteria to match against the loaded model
     * @param device the device the loaded model should use
     * @param progress the progress tracker to update while loading the model
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    ZooModel<I, O> loadModel(Map<String, String> criteria, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException;

    /**
     * Returns a list of the available artifacts that can be loaded.
     *
     * @return a list of the available artifacts that can be loaded
     * @throws IOException for errors reading the artifact list
     * @throws ModelNotFoundException if models with the mrl defined within this loader are found
     */
    List<Artifact> listModels() throws IOException, ModelNotFoundException;
}
