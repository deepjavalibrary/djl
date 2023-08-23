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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.SampleForecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.DeferredTranslatorFactory;
import ai.djl.translate.TranslateException;

import com.google.gson.GsonBuilder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Date;

public final class AirPassengersDeepAR {

    private static final Logger logger = LoggerFactory.getLogger(AirPassengersDeepAR.class);

    private AirPassengersDeepAR() {}

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        float[] results = predict();
        logger.info("{}", results);
    }

    public static float[] predict() throws IOException, TranslateException, ModelException {
        Criteria<TimeSeriesData, Forecast> criteria =
                Criteria.builder()
                        .setTypes(TimeSeriesData.class, Forecast.class)
                        .optModelUrls("djl://ai.djl.mxnet/deepar/0.0.1/airpassengers")
                        .optEngine("MXNet")
                        .optTranslatorFactory(new DeferredTranslatorFactory())
                        .optArgument("prediction_length", 12)
                        .optArgument("freq", "M")
                        .optArgument("use_feat_dynamic_real", false)
                        .optArgument("use_feat_static_cat", false)
                        .optArgument("use_feat_static_real", false)
                        .optProgress(new ProgressBar())
                        .build();

        String url = "https://resources.djl.ai/test-models/mxnet/timeseries/air_passengers.json";

        try (ZooModel<TimeSeriesData, Forecast> model = criteria.loadModel();
                Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor();
                NDManager manager = NDManager.newBaseManager("MXNet")) {
            TimeSeriesData input = getTimeSeriesData(manager, new URL(url));

            // save data for plotting
            NDArray target = input.get(FieldName.TARGET);
            target.setName("target");
            saveNDArray(target);

            Forecast forecast = predictor.predict(input);

            // save data for plotting. Please see the corresponding python script from
            // https://gist.github.com/Carkham/a5162c9298bc51fec648a458a3437008
            NDArray samples = ((SampleForecast) forecast).getSortedSamples();
            samples.setName("samples");
            saveNDArray(samples);
            return forecast.mean().toFloatArray();
        }
    }

    private static TimeSeriesData getTimeSeriesData(NDManager manager, URL url) throws IOException {
        try (Reader reader = new InputStreamReader(url.openStream(), StandardCharsets.UTF_8)) {
            AirPassengers passengers =
                    new GsonBuilder()
                            .setDateFormat("yyyy-MM")
                            .create()
                            .fromJson(reader, AirPassengers.class);

            LocalDateTime start =
                    passengers.start.toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime();
            NDArray target = manager.create(passengers.target);
            TimeSeriesData data = new TimeSeriesData(10);
            data.setStartTime(start);
            data.setField(FieldName.TARGET, target);
            return data;
        }
    }

    private static void saveNDArray(NDArray array) throws IOException {
        Path path = Paths.get("build").resolve(array.getName() + ".npz");
        try (OutputStream os = Files.newOutputStream(path)) {
            new NDList(new NDList(array)).encode(os, NDList.Encoding.NPZ);
        }
    }

    private static final class AirPassengers {

        Date start;
        float[] target;
    }
}
