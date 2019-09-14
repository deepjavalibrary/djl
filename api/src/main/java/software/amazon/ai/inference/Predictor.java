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
package software.amazon.ai.inference;

import java.util.Collections;
import java.util.List;
import software.amazon.ai.Model;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.translate.TranslateException;
import software.amazon.ai.translate.Translator;

/**
 * The {@code Predictor} interface provides model inference functionality.
 *
 * <p>You can use this to do inference with {@link Model} with {@link Translator} specified. The
 * following is example code that uses {@code Predictor}:
 *
 * <pre>
 * Model model = Model.load(modelDir, modelName);
 *
 * // User must implement Translator interface, read Translator for detail.
 * Translator translator = new MyTranslator();
 *
 * try (Predictor&lt;String, String&gt; predictor = <b>model.newPredictor</b>(translator)) {
 *   String result = predictor.<b>predict</b>("What's up");
 * }
 * </pre>
 *
 * @param <I> input type
 * @param <O> output type
 * @see Model
 * @see Translator
 */
public interface Predictor<I, O> extends AutoCloseable {

    /**
     * Predicts an item for inference.
     *
     * @param input Input follows the inputObject
     * @return The Output object defined by user
     * @throws TranslateException if an error occurs during prediction
     */
    default O predict(I input) throws TranslateException {
        return batchPredict(Collections.singletonList(input)).get(0);
    }

    /**
     * Predicts a batch for inference.
     *
     * @param input Inputs follows the inputObject
     * @return The Output objects defined by user
     * @throws TranslateException if an error occurs during prediction
     */
    List<O> batchPredict(List<I> input) throws TranslateException;

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
