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
package ai.djl.gluonTS.translator;

import ai.djl.gluonTS.ForeCast;
import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.SampleForeCast;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.timeFeature.Lag;
import ai.djl.gluonTS.timeFeature.TimeFeature;
import ai.djl.gluonTS.transform.InstanceSampler;
import ai.djl.gluonTS.transform.PredictionSplitSampler;
import ai.djl.gluonTS.transform.convert.Convert;
import ai.djl.gluonTS.transform.feature.Feature;
import ai.djl.gluonTS.transform.field.Field;
import ai.djl.gluonTS.transform.split.Split;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslatorContext;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

/** The {@link ai.djl.translate.Translator} for DeepAR time series forecasting tasks. */
public class DeepARTranslator extends BaseGluonTSTranslator {

    private final boolean useFeatDynamicReal;
    private final boolean useFeatStaticReal;
    private final boolean useFeatStaticCat;

    private int historyLength;
    private int numSamples;

    private static final List<String> PRED_INPUT_FIELDS =
            new ArrayList<String>() {
                {
                    add(FieldName.FEAT_STATIC_CAT.name());
                    add(FieldName.FEAT_STATIC_REAL.name());
                    add("PAST_" + FieldName.FEAT_TIME.name());
                    add("PAST_" + FieldName.TARGET.name());
                    add("PAST_" + FieldName.OBSERVED_VALUES.name());
                    add("FUTURE_" + FieldName.FEAT_TIME.name());
                    add("PAST_" + FieldName.IS_PAD.name());
                }
            };
    private static final List<String> TRAIN_INPUT_FIELDS =
            new ArrayList<String>() {
                {
                    add(FieldName.FEAT_STATIC_CAT.name());
                    add(FieldName.FEAT_STATIC_REAL.name());
                    add("PAST_" + FieldName.FEAT_TIME.name());
                    add("PAST_" + FieldName.TARGET.name());
                    add("PAST_" + FieldName.OBSERVED_VALUES.name());
                    add("PAST_" + FieldName.IS_PAD.name());
                    add("FUTURE" + FieldName.FEAT_TIME.name());
                    add("FUTURE" + FieldName.TARGET.name());
                    add("FUTURE_" + FieldName.OBSERVED_VALUES.name());
                }
            };
    private static final List<FieldName> TIME_SERIES_FIELDS =
            new ArrayList<FieldName>() {
                {
                    add(FieldName.FEAT_TIME);
                    add(FieldName.OBSERVED_VALUES);
                }
            };
    private final List<Integer> lagsSeq;
    private final List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures;

    private final InstanceSampler instanceSampler;

    /**
     * Constructs a {@link DeepARTranslator} with {@link Builder}.
     *
     * @param builder the data to build with
     */
    public DeepARTranslator(Builder builder) {
        super(builder);
        this.useFeatDynamicReal = builder.useFeatDynamicReal;
        this.useFeatStaticReal = builder.useFeatStaticReal;
        this.useFeatStaticCat = builder.useFeatStaticCat;

        this.numSamples = builder.numSamples;

        this.lagsSeq = Lag.getLagsForFreq(freq);
        this.timeFeatures = TimeFeature.timeFeaturesFromFreqStr(freq);
        this.historyLength = contextLength + lagsSeq.get(lagsSeq.size() - 1);
        this.instanceSampler = PredictionSplitSampler.newTestSplitSampler();
    }

    /** {@inheritDoc} */
    @Override
    public ForeCast processOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDArray outputs = list.singletonOrThrow();
        return new SampleForeCast(outputs, this.startTime, this.freq);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, GluonTSData input) throws Exception {
        NDManager manager = ctx.getNDManager();
        this.startTime = input.getStartTime();

        List<FieldName> removeFieldNames = new ArrayList<>();
        removeFieldNames.add(FieldName.FEAT_DYNAMIC_CAT);
        if (!useFeatStaticReal) {
            removeFieldNames.add(FieldName.FEAT_STATIC_REAL);
        }
        if (!useFeatDynamicReal) {
            removeFieldNames.add(FieldName.FEAT_DYNAMIC_REAL);
        }
        input = Field.removeFields(manager, removeFieldNames, input);

        if (!useFeatStaticCat) {
            input =
                    Field.setField(
                            manager, FieldName.FEAT_STATIC_CAT, manager.zeros(new Shape(1)), input);
        }

        if (!useFeatStaticReal) {
            input =
                    Field.setField(
                            manager,
                            FieldName.FEAT_STATIC_REAL,
                            manager.zeros(new Shape(1)),
                            input);
        }

        input =
                Feature.addObservedValuesIndicator(
                        manager, FieldName.TARGET, FieldName.OBSERVED_VALUES, input);

        input =
                Feature.addTimeFeature(
                        manager,
                        FieldName.START,
                        FieldName.TARGET,
                        FieldName.FEAT_TIME,
                        timeFeatures,
                        predictionLength,
                        freq,
                        input);

        input =
                Feature.addAgeFeature(
                        manager, FieldName.TARGET, FieldName.FEAT_AGE, predictionLength, input);

        List<FieldName> inputFields = new ArrayList<>();
        inputFields.add(FieldName.FEAT_TIME);
        inputFields.add(FieldName.FEAT_AGE);
        if (useFeatDynamicReal) {
            inputFields.add(FieldName.FEAT_DYNAMIC_REAL);
        }
        input = Convert.vstackFeatures(manager, FieldName.FEAT_TIME, inputFields, input);

        input =
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

        input = Field.selectField(manager, PRED_INPUT_FIELDS, input);

        return input.toNDList();
    }

    /**
     * Creates a builder to build a {@code DeepARTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code DeepARTranslator}.
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

    /** The builder for DeepAR translator. */
    public static class Builder extends BaseBuilder<Builder> {

        // preProcess args
        private boolean useFeatDynamicReal;
        private boolean useFeatStaticReal;
        private boolean useFeatStaticCat;

        // postProcess args
        private int numSamples;

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
                            arguments, "use_" + FieldName.FEAT_DYNAMIC_REAL.lowerCase(), false);
            this.useFeatStaticCat =
                    ArgumentsUtil.booleanValue(
                            arguments, "use_" + FieldName.FEAT_STATIC_CAT, false);
            this.useFeatStaticReal =
                    ArgumentsUtil.booleanValue(
                            arguments, "use_" + FieldName.FEAT_STATIC_REAL, false);
        }

        /** {@inheritDoc} */
        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);
            this.numSamples = ArgumentsUtil.intValue(arguments, "num_samples", 100);
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public DeepARTranslator build() {
            validate();
            return new DeepARTranslator(this);
        }
    }
}
