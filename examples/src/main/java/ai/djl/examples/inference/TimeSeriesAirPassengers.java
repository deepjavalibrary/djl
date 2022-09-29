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

package ai.djl.examples.inference;

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

import ai.djl.timeseries.translator.DeepARTranslator;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import com.google.gson.GsonBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Arrays;
import java.util.Date;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.GZIPInputStream;

public class TimeSeriesAirPassengers {

    private static final Logger logger = LoggerFactory.getLogger(TimeSeriesAirPassengers.class);

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        float[] results = predict();
        logger.info(Arrays.toString(results));
    }

    public static float[] predict() throws IOException, TranslateException, ModelException {
        String modelUrl = "Not implemented";

        Map<String, Object> arguments = new ConcurrentHashMap<>();
        arguments.put("prediction_length", 12);
        arguments.put("freq", "M");
        arguments.put("use_" + FieldName.FEAT_DYNAMIC_REAL.name().toLowerCase(), false);
        arguments.put("use_" + FieldName.FEAT_STATIC_CAT.name().toLowerCase(), false);
        arguments.put("use_" + FieldName.FEAT_STATIC_REAL.name().toLowerCase(), false);

        DeepARTranslator translator = DeepARTranslator.builder(arguments).build();

        Criteria<TimeSeriesData, Forecast> criteria =
            Criteria.builder()
                .setTypes(TimeSeriesData.class, Forecast.class)
                .optModelPath(Paths.get(modelUrl))
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<TimeSeriesData, Forecast> model = criteria.loadModel();
             Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor()) {
            NDManager manager = model.getNDManager();

            AirPassengers ap =
                new AirPassengers(
                    Paths.get("Not implemented"));
            TimeSeriesData data = ap.get(manager);

            // save data for plotting
            NDArray target = data.get(FieldName.TARGET);
            target.setName("target");
            saveNDArray(target, Paths.get("./target.zip"));

            Forecast forecast = predictor.predict(data);

            NDArray samples = ((SampleForecast) forecast).getSortedSamples();
            samples.setName("samples");
            saveNDArray(samples, Paths.get("./samples.zip"));
            return forecast.mean().toFloatArray();
        }
    }

    public static class AirPassengers {

        private Path path;
        private AirPassengerData data;

        public AirPassengers(Path path) {
            this.path = path;
            prepare();
        }

        public TimeSeriesData get(NDManager manager) {
            LocalDateTime start =
                    data.start.toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime();
            NDArray target = manager.create(data.target);
            TimeSeriesData ret = new TimeSeriesData(10);
            ret.setStartTime(start);
            ret.setField(FieldName.TARGET, target);
            System.out.println(start);
            System.out.println(target.toDebugString(10000, 10000, 10000, 10000));
            return ret;
        }

        private void prepare() {
            Path filePath = path.resolve("test").resolve("data.json.gz");
            try {
                URL url = filePath.toUri().toURL();
                try (GZIPInputStream is = new GZIPInputStream(url.openStream())) {
                    Reader reader = new InputStreamReader(is);
                    data =
                            new GsonBuilder()
                                    .setDateFormat("yyyy-MM")
                                    .create()
                                    .fromJson(reader, AirPassengerData.class);
                }
            } catch (IOException e) {
                throw new IllegalArgumentException("Invalid url: " + filePath, e);
            }
        }

        private static class AirPassengerData {
            Date start;
            float[] target;
        }
    }

    public static void saveNDArray(NDArray array, Path path) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path.toFile())) {
            new NDList(new NDList(array)).encode(fos, true);
        }
    }
}
