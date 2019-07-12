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
package software.amazon.ai;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.Function;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;

/**
 * A model is a collection of artifacts that is created by the training process.
 *
 * <p>A deep learning model usually contains the following parts:
 *
 * <ul>
 *   <li>Graph: aka Symbols in MXNet, model in Keras, Block in Pytorch
 *   <li>Parameters: weights
 *   <li>Input/Output information: input and output parameter names, shape, etc.
 *   <li>Other artifacts: e.g. dictionary for classification
 * </ul>
 *
 * <p>In a common inference case, the model is usually loaded from a file. Once the model is loaded,
 * you can create {@link software.amazon.ai.inference.Predictor} with the loaded model and call
 * {@link software.amazon.ai.inference.Predictor#predict(Object)} to get the inference result.
 *
 * <pre>
 * Model model = <b>Model.loadModel</b>(modelDir, modelName);
 *
 * // User must implement Translator interface, read Translator for detail.
 * Translator translator = new MyTranslator();
 *
 * try (Predictor&lt;String, String&gt; predictor = <b>Predictor.newInstance</b>(model, translator)) {
 *   String result = predictor.<b>predict</b>("What's up");
 * }
 * </pre>
 *
 * @see software.amazon.ai.Model#loadModel(Path, String)
 * @see software.amazon.ai.inference.Predictor
 * @see Translator
 */
public interface Model extends AutoCloseable {

    /**
     * Loads the model from the {@link Path}.
     *
     * @param modelPath path that points to the model file object
     * @return {@link Model} object
     * @throws IOException IO exception happened in loading
     */
    static Model loadModel(Path modelPath) throws IOException {
        return loadModel(modelPath, modelPath.toFile().getName(), null);
    }

    /**
     * Loads the model from the {@link Path} and the given name.
     *
     * @param modelPath Directory/prefix of the file
     * @param modelName model file name or assigned name
     * @return {@link Model} object
     * @throws IOException IO exception happened in loading
     */
    static Model loadModel(Path modelPath, String modelName) throws IOException {
        return loadModel(modelPath, modelName, null);
    }

    /**
     * Loads the model from a {@link Path} object with the name and epoch provided.
     *
     * @param modelPath Directory/prefix of the file
     * @param modelName model file name or assigned name
     * @param options engine specific load model options, see document for each engine
     * @return {@link Model} object
     * @throws IOException IO exception happened in loading
     */
    static Model loadModel(Path modelPath, String modelName, Map<String, String> options)
            throws IOException {
        return Engine.getInstance().loadModel(modelPath, modelName, options);
    }

    /**
     * Returns the input descriptor of the model.
     *
     * <p>It contains the information that can be extracted from the model, usually name, shape,
     * layout and DataType.
     *
     * @return Array of {@link DataDesc}
     */
    DataDesc[] describeInput();

    /**
     * Returns the output descriptor of the model.
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
     * Finds an artifact resource with a given name in the model.
     *
     * @param name name of the desired artifact
     * @return A {@link java.net.URL} object or {@code null} if no artifact with this name is found
     * @throws IOException if an error occurs during loading resource
     */
    URL getArtifact(String name) throws IOException;

    /**
     * Finds an artifact resource with a given name in the model.
     *
     * @param name name of the desired artifact
     * @return A {@link java.io.InputStream} object or {@code null} if no resource with this name is
     *     found
     * @throws IOException if an error occurs during loading resource
     */
    InputStream getArtifactAsStream(String name) throws IOException;

    /**
     * Casts the model to support a different precision level.
     *
     * <p>For example, you can cast the precision from Float to Int
     *
     * @param dataType the target dataType you would like to cast to
     * @return A model with the down casting parameters
     */
    Model cast(DataType dataType);

    /** {@inheritDoc} */
    @Override
    void close();
}
