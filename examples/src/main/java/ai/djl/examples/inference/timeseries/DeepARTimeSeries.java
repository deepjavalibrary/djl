/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference.timeseries;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.timeseries.ForeCast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.translator.DeepARTranslator;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** The example is targeted to specific use case for DeepAR time series forecast. */
public final class DeepARTimeSeries {

    private static final Logger logger = LoggerFactory.getLogger(DeepARTimeSeries.class);

    private DeepARTimeSeries() {}

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        logger.info("model: DeepAR");
        float[] results = DeepARTimeSeries.predict();
        logger.info("Prediction result: {}", Arrays.toString(results));
    }

    public static float[] predict() throws IOException, TranslateException, ModelException {
        String modelUrl = "https://resources.djl.ai/test-models/mxnet/timeseries/deepar.zip";
        Map<String, Object> arguments = new ConcurrentHashMap<>();
        arguments.put("prediction_length", 28);
        DeepARTranslator.Builder builder = DeepARTranslator.builder(arguments);
        DeepARTranslator translator = builder.build();
        Criteria<TimeSeriesData, ForeCast> criteria =
                Criteria.builder()
                        .setTypes(TimeSeriesData.class, ForeCast.class)
                        .optModelUrls(modelUrl)
                        .optTranslator(translator)
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<TimeSeriesData, ForeCast> model = criteria.loadModel();
                Predictor<TimeSeriesData, ForeCast> predictor = model.newPredictor()) {
            TimeSeriesData input = new TimeSeriesData(1);
            input.setStartTime(LocalDateTime.parse("2011-01-29T00:00"));
            NDArray target =
                    model.getNDManager()
                            .randomUniform(0f, 50f, new Shape(1857))
                            .toType(DataType.FLOAT32, false);
            input.setField(FieldName.TARGET, target);
            ForeCast foreCast = predictor.predict(input);

            return foreCast.mean().toFloatArray();
        }
    }
}
