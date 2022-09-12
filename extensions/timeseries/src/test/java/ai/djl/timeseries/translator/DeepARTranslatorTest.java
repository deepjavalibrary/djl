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
package ai.djl.timeseries.translator;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestRequirements;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class DeepARTranslatorTest {

    @Test
    public void testDeepARTranslator() throws IOException, TranslateException, ModelException {
        TestRequirements.notArm();

        String modelUrl = "https://resources.djl.ai/test-models/mxnet/timeseries/deepar.zip";
        Map<String, Object> arguments = new ConcurrentHashMap<>();
        arguments.put("prediction_length", 28);
        DeepARTranslator.Builder builder = DeepARTranslator.builder(arguments);
        DeepARTranslator translator = builder.build();
        Criteria<TimeSeriesData, Forecast> criteria =
                Criteria.builder()
                        .setTypes(TimeSeriesData.class, Forecast.class)
                        .optModelUrls(modelUrl)
                        .optTranslator(translator)
                        .optProgress(new ProgressBar())
                        .build();

        try (NDManager manager = NDManager.newBaseManager()) {
            manager.getEngine().setRandomSeed(1);
            NDArray target = manager.arange(0.0f, 50.0f, (float) 50 / 1856);

            TimeSeriesData input = new TimeSeriesData(1);
            input.setStartTime(LocalDateTime.parse("2011-01-29T00:00"));
            input.setField(FieldName.TARGET, target);

            try (ZooModel<TimeSeriesData, Forecast> model = criteria.loadModel();
                    Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor()) {
                Forecast forecast = predictor.predict(input);
                // TODO： The argument prediction_length = 28 is too far away from predict().
                // Here forecast.mean() is a predicted sequence of length = "prediction_length" set
                // above. That forecast.mean() still has randomness here is because the model
                // imported here was trained on a sparse dataSet with many zeros (inactive sale
                // amount). So here the model also predict for such inactive data once in a while
                // interweaving the active data.
                // TODO: Either preprocess the data to choose a proper time spacing to reduce the
                // sparsity of the sequence ie the inactive numbers (ie. 0).
                Assert.assertEquals(forecast.mean().toFloatArray().length, 28);
            }
        }
    }
}
