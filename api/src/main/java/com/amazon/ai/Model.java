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
package com.amazon.ai;

import com.amazon.ai.engine.Engine;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Function;

/**
 * The <code>Model</code> interface is the holder of the model.
 *
 * <p>Users can use this to load the model and apply it for {@link com.amazon.ai.training.Trainer}
 * and {@link com.amazon.ai.inference.Predictor} for Training and Inference jobs.
 */
public interface Model {

    /**
     * Load the model from a file path with epoch provided.
     *
     * <p>It will try to find the model like res-152-0002.param
     *
     * @param modelPath Path to the model, include the model name
     * @param epoch number of epoch of the model
     * @return {@link Model} object
     * @throws IOException IO exception happened in loading
     */
    static Model loadModel(String modelPath, int epoch) throws IOException {
        Path path = Paths.get(modelPath);
        String modelName = path.toFile().getName();
        return loadModel(path, modelName, epoch);
    }

    /**
     * Load the model from the {@link Path}.
     *
     * @param modelPath File object point to a path
     * @return {@link Model} object
     * @throws IOException IO exception happened in loading
     */
    static Model loadModel(Path modelPath) throws IOException {
        return loadModel(modelPath, modelPath.toFile().getName(), -1);
    }

    /**
     * Load the model from the {@link Path} and the given name.
     *
     * @param modelPath Directory/prefix of the file
     * @param modelName model file name or assigned name
     * @return {@link Model} object
     * @throws IOException IO exception happened in loading
     */
    static Model loadModel(Path modelPath, String modelName) throws IOException {
        return loadModel(modelPath, modelName, -1);
    }

    /**
     * Load the model from a {@link Path} object with name and epoch provided.
     *
     * @param modelPath Directory/prefix of the file
     * @param modelName model file name or assigned name
     * @param epoch number of epoch of the model
     * @return {@link Model} object
     * @throws IOException IO exception happened in loading
     */
    static Model loadModel(Path modelPath, String modelName, int epoch) throws IOException {
        return Engine.getInstance().loadModel(modelPath, modelName, epoch);
    }
    /**
     * Get the input descriptor of the model.
     *
     * <p>It contains the information that can be extracted from the model, usually name, shape,
     * layout and DataType.
     *
     * @return Array of {@link DataDesc}
     */
    DataDesc[] describeInput();

    /**
     * Get the output descriptor of the model.
     *
     * <p>It contains the output information that can be obtained from the model
     *
     * @return Array of {@link DataDesc}
     */
    DataDesc[] describeOutput();

    /**
     * Returns artifact names associated with the model.
     *
     * @return array of artifact names
     */
    String[] getArtifactNames();

    /**
     * If the specified artifact is not already cached, attempts to load the artifact using the
     * given function and cache it.
     *
     * <p>Model will cache loaded artifact, so user doesn't need to keep tracking it.
     *
     * <pre>{@code
     * String synset = model.getArtifact("synset.txt", k -> IOUtils.toString(k)));
     * }</pre>
     *
     * @param name name of the desired artifact
     * @param function the function to load artifact
     * @param <T> type of return artifact object
     * @return the current (existing or computed) artifact associated with the specified name, or
     *     null if the computed value is null
     * @throws IOException if an error occurs during loading resource
     * @throws ClassCastException if the cached artifact cannot be casted to target class
     */
    <T> T getArtifact(String name, Function<InputStream, T> function) throws IOException;

    /**
     * Finds a artifact resource with a given name in the model.
     *
     * @param name name of the desired artifact
     * @return A {@link java.net.URL} object or {@code null} if no artifact with this name is found
     * @throws IOException if an error occurs during loading resource
     */
    URL getArtifact(String name) throws IOException;

    /**
     * Finds a artifact resource with a given name in the model.
     *
     * @param name name of the desired artifact
     * @return A {@link java.io.InputStream} object or {@code null} if no resource with this name is
     *     found
     * @throws IOException if an error occurs during loading resource
     */
    InputStream getArtifactAsStream(String name) throws IOException;

    /**
     * Cast the model to support different precision level.
     *
     * <p>For example, you can cast the precision from Float to Int
     *
     * @param dataType the target dataType you would like to cast to
     * @return A model with the down casting parameters
     */
    Model cast(DataType dataType);
}
