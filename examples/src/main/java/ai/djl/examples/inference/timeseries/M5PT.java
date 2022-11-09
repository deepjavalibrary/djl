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

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.tabular.utils.DynamicBuffer;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.SampleForecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.DeferredTranslatorFactory;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class M5PT {

    private static final Logger logger = LoggerFactory.getLogger(M5PT.class);

    private M5PT() {}

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        Map<String, Float> metrics = predict();
        for (Map.Entry<String, Float> entry : metrics.entrySet()) {
            logger.info("{}", String.format("metric: %s:\t%.2f", entry.getKey(), entry.getValue()));
        }
    }

    public static Map<String, Float> predict()
            throws IOException, TranslateException, ModelException {
        NDManager manager = NDManager.newBaseManager(null, "MXNet");

        // To use local dataset, users can load data as follows
        // Repository repository = Repository.newInstance("local_dataset",
        // Paths.get("rootPath/m5-forecasting-accuracy"));
        // Then add the setting `.optRepository(repository)` to the builder below
        M5Dataset dataset = M5Dataset.builder().setManager(manager).build();

        // The modelUrl can be replaced by local model path. E.g.,
        // String modelUrl = "rootPath/deepar.zip";
        Path modelPath = Paths.get("./src/main/resources/deepar.pt");
        int predictionLength = 4;
        Criteria<TimeSeriesData, Forecast> criteria =
                Criteria.builder()
                        .setTypes(TimeSeriesData.class, Forecast.class)
                        .optModelPath(modelPath)
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new DeferredTranslatorFactory())
                        .optArgument("prediction_length", predictionLength)
                        .optArgument("freq", "W")
                        .optArgument("use_feat_dynamic_real", "false")
                        .optArgument("use_feat_static_cat", "false")
                        .optArgument("use_feat_static_real", "false")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<TimeSeriesData, Forecast> model = criteria.loadModel();
                Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor()) {
            M5Evaluator evaluator = new M5Evaluator(0.5f, 0.67f, 0.95f, 0.99f);
            Progress progress = new ProgressBar();
            progress.reset("Inferring", dataset.size);
            for (NDList data : dataset) {
                NDArray array = data.singletonOrThrow();
                NDArray gt = array.get("{}:", -predictionLength);
                NDArray pastTarget = array.get(":{}", -predictionLength);
                TimeSeriesData input = new TimeSeriesData(10);
                input.setStartTime(LocalDateTime.parse("2011-01-29T00:00"));
                input.setField(FieldName.TARGET, pastTarget);
                Forecast forecast = predictor.predict(input);
                // We focus on the metric Weighted Root Mean Squared Scaled Error (RMSSE) same as
                // https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview/evaluation
                // The error is not small compared to the data values (sale amount). This is because
                // The model is trained on a sparse data with many zeros. This will be improved by
                // aggregating/coarse graining the data. See https://github.com/Carkham/m5_blog
                evaluator.aggregateMetrics(evaluator.getMetricsPerTs(gt, pastTarget, forecast));
                progress.increment(1);

                // save data for plotting. Please see the corresponding python script from
                // https://gist.github.com/Carkham/a5162c9298bc51fec648a458a3437008
                NDArray samples = ((SampleForecast) forecast).getSortedSamples();
                samples.setName("samples");
                saveNDArray(samples);
            }

            manager.close();
            return evaluator.computeTotalMetrics();
        }
    }

    private static void saveNDArray(NDArray array) throws IOException {
        Path path = Paths.get("build").resolve(array.getName() + ".npz");
        try (OutputStream os = Files.newOutputStream(path)) {
            new NDList(new NDList(array)).encode(os, true);
        }
    }

    /**
     * M5 Forecasting - Accuracy from <a
     * href="https://www.kaggle.com/competitions/m5-forecasting-accuracy">https://www.kaggle.com/competitions/m5-forecasting-accuracy</a>
     *
     * <p>Each csvRecord contains a target from "d_1" to "d_1941".
     */
    private static final class M5Dataset implements Iterable<NDList>, Iterator<NDList> {

        private NDManager manager;
        private List<Feature> target;
        private List<CSVRecord> csvRecords;
        private long size;
        private long current;

        M5Dataset(Builder builder) {
            manager = builder.manager;
            target = builder.target;
            try {
                prepare(builder);
            } catch (Exception e) {
                throw new AssertionError("Failed to read files.", e);
            }
            size = csvRecords.size();
        }

        private void prepare(Builder builder) throws IOException {
            MRL mrl = builder.getMrl();
            Artifact artifact = mrl.getDefaultArtifact();
            mrl.prepare(artifact, null);

            Path root = mrl.getRepository().getResourceDirectory(artifact);
            Path csvFile = root.resolve("weekly_sales_train_evaluation.csv");

            URL csvUrl = csvFile.toUri().toURL();
            try (Reader reader =
                    new InputStreamReader(
                            new BufferedInputStream(csvUrl.openStream()), StandardCharsets.UTF_8)) {
                CSVParser csvParser = new CSVParser(reader, builder.csvFormat);
                csvRecords = csvParser.getRecords();
            }
        }

        @Override
        public boolean hasNext() {
            return current < size;
        }

        @Override
        public NDList next() {
            NDList data = getRowFeatures(manager, current, target);
            current++;
            return data;
        }

        public static Builder builder() {
            return new Builder();
        }

        private NDList getRowFeatures(NDManager manager, long index, List<Feature> selected) {
            DynamicBuffer bb = new DynamicBuffer();
            for (Feature feature : selected) {
                String name = feature.getName();
                String value = getCell(index, name);
                feature.getFeaturizer().featurize(bb, value);
            }
            FloatBuffer buf = bb.getBuffer();
            return new NDList(manager.create(buf, new Shape(bb.getLength())));
        }

        private String getCell(long rowIndex, String featureName) {
            CSVRecord record = csvRecords.get(Math.toIntExact(rowIndex));
            return record.get(featureName);
        }

        @Override
        public Iterator<NDList> iterator() {
            return this;
        }

        public static final class Builder {

            NDManager manager;
            List<Feature> target;
            CSVFormat csvFormat;

            Repository repository;
            String groupId;
            String artifactId;
            String version;

            Builder() {
                repository = BasicDatasets.REPOSITORY;
                groupId = BasicDatasets.GROUP_ID;
                artifactId = "m5forecast-unittest";
                version = "1.0";
                csvFormat =
                        CSVFormat.DEFAULT
                                .builder()
                                .setHeader()
                                .setSkipHeaderRecord(true)
                                .setIgnoreHeaderCase(true)
                                .setTrim(true)
                                .build();
                target = new ArrayList<>();
                for (int i = 1; i <= 277; i++) {
                    target.add(new Feature("w_" + i, true));
                }
            }

            public Builder optRepository(Repository repository) {
                this.repository = repository;
                return this;
            }

            public Builder setManager(NDManager manager) {
                this.manager = manager;
                return this;
            }

            public M5Dataset build() {
                return new M5Dataset(this);
            }

            MRL getMrl() {
                return repository.dataset(Application.Tabular.ANY, groupId, artifactId, version);
            }
        }
    }

    /** An evaluator that calculates performance metrics. */
    public static final class M5Evaluator {
        private float[] quantiles;
        Map<String, Float> totalMetrics;
        Map<String, Integer> totalNum;

        public M5Evaluator(float... quantiles) {
            this.quantiles = quantiles;
            totalMetrics = new ConcurrentHashMap<>();
            totalNum = new ConcurrentHashMap<>();
            init();
        }

        public Map<String, Float> getMetricsPerTs(
                NDArray gtTarget, NDArray pastTarget, Forecast forecast) {
            Map<String, Float> retMetrics =
                    new ConcurrentHashMap<>((8 + quantiles.length * 2) * 3 / 2);
            NDArray meanFcst = forecast.mean();
            NDArray medianFcst = forecast.median();

            NDArray meanSquare = gtTarget.sub(meanFcst).square().mean();
            NDArray scaleDenom = gtTarget.get("1:").sub(gtTarget.get(":-1")).square().mean();

            NDArray rmsse = meanSquare.div(scaleDenom).sqrt();
            rmsse = NDArrays.where(scaleDenom.eq(0), rmsse.onesLike(), rmsse);

            retMetrics.put("RMSSE", rmsse.getFloat());

            retMetrics.put("MSE", gtTarget.sub(meanFcst).square().mean().getFloat());
            retMetrics.put("abs_error", gtTarget.sub(medianFcst).abs().sum().getFloat());
            retMetrics.put("abs_target_sum", gtTarget.abs().sum().getFloat());
            retMetrics.put("abs_target_mean", gtTarget.abs().mean().getFloat());
            retMetrics.put(
                    "MAPE", gtTarget.sub(medianFcst).abs().div(gtTarget.abs()).mean().getFloat());
            retMetrics.put(
                    "sMAPE",
                    gtTarget.sub(medianFcst)
                            .abs()
                            .div(gtTarget.abs().add(medianFcst.abs()))
                            .mean()
                            .mul(2)
                            .getFloat());
            retMetrics.put("ND", retMetrics.get("abs_error") / retMetrics.get("abs_target_sum"));

            for (float quantile : quantiles) {
                NDArray forecastQuantile = forecast.quantile(quantile);

                NDArray quantileLoss =
                        forecastQuantile
                                .sub(gtTarget)
                                .mul(gtTarget.lte(forecastQuantile).sub(quantile))
                                .abs()
                                .sum()
                                .mul(2);
                NDArray quantileCoverage = gtTarget.lt(forecastQuantile).mean();
                retMetrics.put(
                        String.format("QuantileLoss[%.2f]", quantile), quantileLoss.getFloat());
                retMetrics.put(
                        String.format("Coverage[%.2f]", quantile), quantileCoverage.getFloat());
            }
            return retMetrics;
        }

        public void aggregateMetrics(Map<String, Float> metrics) {
            for (Map.Entry<String, Float> entry : metrics.entrySet()) {
                totalMetrics.compute(entry.getKey(), (k, v) -> v + entry.getValue());
                totalNum.compute(entry.getKey(), (k, v) -> v + 1);
            }
        }

        public Map<String, Float> computeTotalMetrics() {
            for (Map.Entry<String, Integer> entry : totalNum.entrySet()) {
                if (!entry.getKey().contains("sum")) {
                    totalMetrics.compute(entry.getKey(), (k, v) -> v / (float) entry.getValue());
                }
            }

            totalMetrics.put("RMSE", (float) Math.sqrt(totalMetrics.get("MSE")));
            totalMetrics.put(
                    "NRMSE", totalMetrics.get("RMSE") / totalMetrics.get("abs_target_mean"));
            return totalMetrics;
        }

        private void init() {
            List<String> metricNames =
                    new ArrayList<>(
                            Arrays.asList(
                                    "RMSSE",
                                    "MSE",
                                    "abs_error",
                                    "abs_target_sum",
                                    "abs_target_mean",
                                    "MAPE",
                                    "sMAPE",
                                    "ND"));
            for (float quantile : quantiles) {
                metricNames.add(String.format("QuantileLoss[%.2f]", quantile));
                metricNames.add(String.format("Coverage[%.2f]", quantile));
            }
            for (String metricName : metricNames) {
                totalMetrics.put(metricName, 0f);
                totalNum.put(metricName, 0);
            }
        }
    }
}
