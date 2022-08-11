package ai.djl.gluonTS.translator;

import ai.djl.gluonTS.ForeCast;
import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.timeFeature.Lag;
import ai.djl.gluonTS.timeFeature.TimeFeature;
import ai.djl.gluonTS.transform.*;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslatorContext;

import java.time.LocalDateTime;
import java.util.*;
import java.util.function.BiFunction;

/** The {@link ai.djl.translate.Translator} for Transformer time series forecasting tasks. */
public class TransformerTranslator extends BaseGluonTSTranslator {

    private final boolean useFeatDynamicReal;
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
                    add("FUTURE" + FieldName.FEAT_TIME.name());
                    add("FUTURE" + FieldName.TARGET.name());
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

    public TransformerTranslator(Builder builder) {
        super(builder);
        this.useFeatDynamicReal = builder.useFeatDynamicReal;
        this.useFeatStaticCat = builder.useFeatStaticCat;

        this.numSamples = builder.numSamples;

        this.lagsSeq = Lag.getLagsForFreq(freq);
        this.timeFeatures = TimeFeature.timeFeaturesFromFreqStr(freq);
        this.historyLength = context_length + lagsSeq.get(lagsSeq.size() - 1);
        this.instanceSampler = PredictionSplitSampler.newTestSplitSampler();
    }

    /** {@inheritDoc} */
    @Override
    public ForeCast processOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDArray outputs = list.singletonOrThrow();
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, GluonTSData input) throws Exception {
        NDManager manager = ctx.getNDManager();

        List<FieldName> removeFieldNames = new ArrayList<>();
        removeFieldNames.add(FieldName.FEAT_DYNAMIC_CAT);
        removeFieldNames.add(FieldName.FEAT_STATIC_REAL);

        if (!useFeatDynamicReal) {
            removeFieldNames.add(FieldName.FEAT_DYNAMIC_REAL);
        }
        input = Field.removeFields(manager, removeFieldNames, input);

        if (!useFeatStaticCat) {
            input =
                    Field.setField(
                            manager, FieldName.FEAT_STATIC_CAT, manager.zeros(new Shape(1)), input);
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

    /** The builder for Transformer translator. */
    public static class Builder extends BaseBuilder<Builder> {

        // preProcess args
        private boolean useFeatDynamicReal;
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
        public TransformerTranslator build() {
            validate();
            return new TransformerTranslator(this);
        }
    }
}
