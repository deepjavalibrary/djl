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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.SampleForecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.timefeature.Lag;
import ai.djl.timeseries.timefeature.TimeFeature;
import ai.djl.timeseries.transform.InstanceSampler;
import ai.djl.timeseries.transform.PredictionSplitSampler;
import ai.djl.timeseries.transform.convert.Convert;
import ai.djl.timeseries.transform.feature.Feature;
import ai.djl.timeseries.transform.field.Field;
import ai.djl.timeseries.transform.split.Split;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

/** The {@link Translator} for Transformer time series forecasting tasks. */
public class TransformerTranslator extends BaseTimeSeriesTranslator {

    private final boolean useFeatDynamicReal;
    private final boolean useFeatStaticCat;

    private int historyLength;

    private static final List<String> PRED_INPUT_FIELDS =
            new ArrayList<>(
                    Arrays.asList(
                            FieldName.FEAT_STATIC_CAT.name(),
                            "PAST_" + FieldName.FEAT_TIME.name(),
                            "PAST_" + FieldName.TARGET.name(),
                            "PAST_" + FieldName.OBSERVED_VALUES.name(),
                            "FUTURE_" + FieldName.FEAT_TIME.name()));
    private static final List<FieldName> TIME_SERIES_FIELDS =
            new ArrayList<>(Arrays.asList(FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES));
    private final List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures;

    private final InstanceSampler instanceSampler;

    /**
     * Constructs a {@link TransformerTranslator} with {@link Builder}.
     *
     * @param builder the data to build with
     */
    public TransformerTranslator(Builder builder) {
        super(builder);
        this.useFeatDynamicReal = builder.useFeatDynamicReal;
        this.useFeatStaticCat = builder.useFeatStaticCat;

        List<Integer> lagsSeq = Lag.getLagsForFreq(freq);
        this.timeFeatures = TimeFeature.timeFeaturesFromFreqStr(freq);
        this.historyLength = contextLength + lagsSeq.get(lagsSeq.size() - 1);
        this.instanceSampler = PredictionSplitSampler.newTestSplitSampler();
    }

    /** {@inheritDoc} */
    @Override
    public Forecast processOutput(TranslatorContext ctx, NDList list) {
        NDArray outputs = list.singletonOrThrow();
        return new SampleForecast(outputs, this.startTime, this.freq);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, TimeSeriesData input) {
        NDManager manager = ctx.getNDManager();
        this.startTime = input.getStartTime();

        List<FieldName> removeFieldNames = new ArrayList<>();
        removeFieldNames.add(FieldName.FEAT_DYNAMIC_CAT);
        removeFieldNames.add(FieldName.FEAT_STATIC_REAL);

        if (!useFeatDynamicReal) {
            removeFieldNames.add(FieldName.FEAT_DYNAMIC_REAL);
        }
        Field.removeFields(removeFieldNames, input);

        if (!useFeatStaticCat) {
            Field.setField(FieldName.FEAT_STATIC_CAT, manager.zeros(new Shape(1)), input);
        }

        Feature.addObservedValuesIndicator(
                manager, FieldName.TARGET, FieldName.OBSERVED_VALUES, input);

        Feature.addTimeFeature(
                manager,
                FieldName.START,
                FieldName.TARGET,
                FieldName.FEAT_TIME,
                timeFeatures,
                predictionLength,
                freq,
                input);

        Feature.addAgeFeature(
                manager, FieldName.TARGET, FieldName.FEAT_AGE, predictionLength, input);

        List<FieldName> inputFields = new ArrayList<>();
        inputFields.add(FieldName.FEAT_TIME);
        inputFields.add(FieldName.FEAT_AGE);
        if (useFeatDynamicReal) {
            inputFields.add(FieldName.FEAT_DYNAMIC_REAL);
        }
        Convert.vstackFeatures(FieldName.FEAT_TIME, inputFields, input);

        Split.instanceSplit(
                manager,
                FieldName.TARGET,
                FieldName.IS_PAD,
                FieldName.START,
                FieldName.FORECAST_START,
                instanceSampler,
                historyLength,
                predictionLength,
                TIME_SERIES_FIELDS,
                0,
                input);

        input = Field.selectField(PRED_INPUT_FIELDS, input);

        return input.toNDList();
    }

    /**
     * Creates a builder to build a {@code TransformerTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code TransformerTranslator}.
     *
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = new Builder();

        builder.configPreProcess(arguments);
        builder.configPostProcess(arguments);

        return builder;
    }

    /** The builder for Transformer translator. */
    public static class Builder extends BaseBuilder<Builder> {

        // preProcess args
        private boolean useFeatDynamicReal;
        private boolean useFeatStaticCat;

        // postProcess args

        Builder() {}

        @Override
        protected Builder self() {
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected void configPreProcess(Map<String, ?> arguments) {
            super.configPreProcess(arguments);
            this.useFeatDynamicReal =
                    ArgumentsUtil.booleanValue(
                            arguments,
                            "use_" + FieldName.FEAT_DYNAMIC_REAL.name().toLowerCase(),
                            false);
            this.useFeatStaticCat =
                    ArgumentsUtil.booleanValue(
                            arguments,
                            "use_" + FieldName.FEAT_STATIC_CAT.name().toLowerCase(),
                            false);
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public TransformerTranslator build() {
            validate();
            return new TransformerTranslator(this);
        }
    }
}
