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
package ai.djl.inference;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import java.util.List;

/**
 * The {@code Predictor} interface provides model inference functionality.
 *
 * <p>You can use a {@code Predictor}, with a specified {@link Translator}, to perform inference on
 * a {@link Model}. The following is example code that uses {@code Predictor}:
 *
 * <pre>
 * Model model = Model.load(modelDir, modelName);
 *
 * // User must implement Translator interface, read {@link Translator} for detail.
 * Translator translator = new MyTranslator();
 *
 * try (Predictor&lt;String, String&gt; predictor = model.newPredictor(translator)) {
 *   String result = predictor.predict("What's up");
 * }
 * </pre>
 *
 * <p>See the tutorials on:
 *
 * <ul>
 *   <li><a
 *       href="https://github.com/awslabs/djl/blob/master/jupyter/image_classification_with_your_model.ipynb">Inference
 *       with a custom trained model</a>
 *   <li><a
 *       href="https://github.com/awslabs/djl/blob/master/jupyter/object_detection_with_model_zoo.ipynb">Inference
 *       with a model zoo model</a>
 *   <li><a
 *       href="https://github.com/awslabs/djl/blob/master/jupyter/load_mxnet_model.ipynb">Inference
 *       with an MXNet model</a>
 * </ul>
 *
 * @param <I> the input type
 * @param <O> the output type
 * @see Model
 * @see Translator
 */
public interface Predictor<I, O> extends AutoCloseable {

    /**
     * Predicts an item for inference.
     *
     * @param input the input
     * @return the output object defined by the user
     * @throws TranslateException if an error occurs during prediction
     */
    O predict(I input) throws TranslateException;

    /**
     * Predicts a batch for inference.
     *
     * @param inputs a list of inputs
     * @return a list of output objects defined by the user
     * @throws TranslateException if an error occurs during prediction
     */
    List<O> batchPredict(List<I> inputs) throws TranslateException;

    /**
     * Attaches a Metrics param to use for benchmark.
     *
     * @param metrics the Metrics class
     */
    void setMetrics(Metrics metrics);

    /** {@inheritDoc} */
    @Override
    void close();
}
