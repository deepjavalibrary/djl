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

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.repository.Artifact;
import java.io.IOException;
import java.util.List;

/** A ModelLoader loads a particular {@link ZooModel} from a Repository for a model zoo. */
public interface ModelLoader {

    /**
     * Returns the artifact ID of the {@code ModelLoader}.
     *
     * @return the artifact ID of the {@code ModelLoader}
     */
    String getArtifactId();

    /**
     * Returns the application of the {@code ModelLoader}.
     *
     * @return the application of the {@code ModelLoader}
     */
    Application getApplication();

    /**
     * Loads the model with the given criteria.
     *
     * @param <I> the input data type
     * @param <O> the output data type
     * @param criteria the criteria to match against the loaded model
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    <I, O> ZooModel<I, O> loadModel(Criteria<I, O> criteria)
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
