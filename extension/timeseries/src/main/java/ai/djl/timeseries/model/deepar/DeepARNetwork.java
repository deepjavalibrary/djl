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

package ai.djl.timeseries.model.deepar;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.timeseries.block.FeatureEmbedder;
import ai.djl.timeseries.block.MeanScaler;
import ai.djl.timeseries.block.NopScaler;
import ai.djl.timeseries.block.Scaler;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.timeseries.distribution.output.StudentTOutput;
import ai.djl.timeseries.timefeature.Lag;
import ai.djl.timeseries.timefeature.TimeFeature;
import ai.djl.timeseries.transform.ExpectedNumInstanceSampler;
import ai.djl.timeseries.transform.InstanceSampler;
import ai.djl.timeseries.transform.PredictionSplitSampler;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.timeseries.transform.convert.AsArray;
import ai.djl.timeseries.transform.convert.VstackFeatures;
import ai.djl.timeseries.transform.feature.AddAgeFeature;
import ai.djl.timeseries.transform.feature.AddObservedValuesIndicator;
import ai.djl.timeseries.transform.feature.AddTimeFeature;
import ai.djl.timeseries.transform.field.RemoveFields;
import ai.djl.timeseries.transform.field.SelectField;
import ai.djl.timeseries.transform.field.SetField;
import ai.djl.timeseries.transform.split.InstanceSplit;
import ai.djl.training.ParameterStore;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Implements the deepar model.
 *
 * <p>This closely follows the <a
 * href="https://linkinghub.elsevier.com/retrieve/pii/S0169207019301888">Salinas et al. 2020</a> and
 * its <a href="https://github.com/awslabs/gluonts">gluonts</a> implementation.
 */
public abstract class DeepARNetwork extends AbstractBlock {

    private static final String[] TRAIN_INPUT_FIELDS = {
        FieldName.FEAT_STATIC_CAT.name(),
        FieldName.FEAT_STATIC_REAL.name(),
        "PAST_" + FieldName.FEAT_TIME.name(),
        "PAST_" + FieldName.TARGET.name(),
        "PAST_" + FieldName.OBSERVED_VALUES.name(),
        "PAST_" + FieldName.IS_PAD.name(),
        "FUTURE_" + FieldName.FEAT_TIME.name(),
        "FUTURE_" + FieldName.TARGET.name(),
        "FUTURE_" + FieldName.OBSERVED_VALUES.name()
    };

    private static final String[] PRED_INPUT_FIELDS = {
        FieldName.FEAT_STATIC_CAT.name(),
        FieldName.FEAT_STATIC_REAL.name(),
        "PAST_" + FieldName.FEAT_TIME.name(),
        "PAST_" + FieldName.TARGET.name(),
        "PAST_" + FieldName.OBSERVED_VALUES.name(),
        "FUTURE_" + FieldName.FEAT_TIME.name(),
        "PAST_" + FieldName.IS_PAD.name()
    };

    protected String freq;
    protected int historyLength;
    protected int contextLength;
    protected int predictionLength;

    protected boolean useFeatDynamicReal;
    protected boolean useFeatStaticCat;
    protected boolean useFeatStaticReal;

    protected DistributionOutput distrOutput;
    protected List<Integer> cardinality;
    protected List<Integer> embeddingDimension;
    protected List<Integer> lagsSeq;
    protected int numParallelSamples;

    protected FeatureEmbedder embedder;
    protected Block paramProj;

    protected LSTM rnn;
    protected Scaler scaler;

    DeepARNetwork(Builder builder) {
        freq = builder.freq;
        predictionLength = builder.predictionLength;
        contextLength = builder.contextLength != 0 ? builder.contextLength : predictionLength;
        distrOutput = builder.distrOutput;
        cardinality = builder.cardinality;

        useFeatStaticReal = builder.useFeatStaticReal;
        useFeatDynamicReal = builder.useFeatDynamicReal;
        useFeatStaticCat = builder.useFeatStaticCat;
        numParallelSamples = builder.numParallelSamples;

        paramProj = addChildBlock("param_proj", distrOutput.getArgsProj());
        if (builder.embeddingDimension != null || builder.cardinality == null) {
            embeddingDimension = builder.embeddingDimension;
        } else {
            embeddingDimension = new ArrayList<>();
            for (int cat : cardinality) {
                embeddingDimension.add(Math.min(50, (cat + 1) / 2));
            }
        }
        lagsSeq = builder.lagsSeq == null ? Lag.getLagsForFreq(builder.freq) : builder.lagsSeq;
        historyLength = contextLength + lagsSeq.stream().max(Comparator.naturalOrder()).get();
        embedder =
                addChildBlock(
                        "feature_embedder",
                        FeatureEmbedder.builder()
                                .setCardinalities(cardinality)
                                .setEmbeddingDims(embeddingDimension)
                                .build());
        if (builder.scaling) {
            scaler =
                    addChildBlock(
                            "scaler",
                            MeanScaler.builder()
                                    .setDim(1)
                                    .optKeepDim(true)
                                    .optMinimumScale(1e-10f)
                                    .build());
        } else {
            scaler =
                    addChildBlock("scaler", NopScaler.builder().setDim(1).optKeepDim(true).build());
        }

        rnn =
                addChildBlock(
                        "rnn_lstm",
                        LSTM.builder()
                                .setNumLayers(builder.numLayers)
                                .setStateSize(builder.hiddenSize)
                                .optDropRate(builder.dropRate)
                                .optBatchFirst(true)
                                .optReturnState(true)
                                .build());
    }

    /** {@inheritDoc} */
    @Override
    protected void initializeChildBlocks(
            NDManager manager, DataType dataType, Shape... inputShapes) {

        Shape targetShape = inputShapes[3].slice(2);
        Shape contextShape = new Shape(1, contextLength).addAll(targetShape);
        scaler.initialize(manager, dataType, contextShape, contextShape);
        long scaleSize = scaler.getOutputShapes(new Shape[] {contextShape, contextShape})[1].get(1);

        embedder.initialize(manager, dataType, inputShapes[0]);
        long embeddedCatSize = embedder.getOutputShapes(new Shape[] {inputShapes[0]})[0].get(1);

        Shape inputShape = new Shape(1, contextLength * 2L - 1).addAll(targetShape);
        Shape lagsShape = inputShape.add(lagsSeq.size());
        long featSize = inputShapes[2].get(2) + embeddedCatSize + inputShapes[1].get(1) + scaleSize;
        Shape rnnInputShape =
                lagsShape.slice(0, lagsShape.dimension() - 1).add(lagsShape.tail() + featSize);
        rnn.initialize(manager, dataType, rnnInputShape);

        Shape rnnOutShape = rnn.getOutputShapes(new Shape[] {rnnInputShape})[0];
        paramProj.initialize(manager, dataType, rnnOutShape);
    }

    /**
     * Applies the underlying RNN to the provided target data and covariates.
     *
     * @param ps the parameter store
     * @param inputs the input NDList
     * @param training true for a training forward pass
     * @return a {@link NDList} containing arguments of the output distribution, scaling factor, raw
     *     output of rnn, static input of rnn, output state of rnn
     */
    protected NDList unrollLaggedRnn(ParameterStore ps, NDList inputs, boolean training) {
        try (NDManager scope = inputs.getManager().newSubManager()) {
            scope.tempAttachAll(inputs);

            NDArray featStaticCat = inputs.get(0);
            NDArray featStaticReal = inputs.get(1);
            NDArray pastTimeFeat = inputs.get(2);
            NDArray pastTarget = inputs.get(3);
            NDArray pastObservedValues = inputs.get(4);
            NDArray futureTimeFeat = inputs.get(5);
            NDArray futureTarget = inputs.size() > 6 ? inputs.get(6) : null;

            NDArray context = pastTarget.get(":,{}:", -contextLength);
            NDArray observedContext = pastObservedValues.get(":,{}:", -contextLength);
            NDArray scale =
                    scaler.forward(ps, new NDList(context, observedContext), training).get(1);

            NDArray priorSequence = pastTarget.get(":,:{}", -contextLength).div(scale);
            NDArray sequence =
                    futureTarget != null
                            ? context.concat(futureTarget.get(":, :-1"), 1).div(scale)
                            : context.div(scale);

            NDArray embeddedCat =
                    embedder.forward(ps, new NDList(featStaticCat), training).singletonOrThrow();
            NDArray staticFeat =
                    NDArrays.concat(new NDList(embeddedCat, featStaticReal, scale.log()), 1);
            NDArray expandedStaticFeat =
                    staticFeat.expandDims(1).repeat(1, sequence.getShape().get(1));

            NDArray timeFeat =
                    futureTimeFeat != null
                            ? pastTimeFeat
                                    .get(":, {}:", -contextLength + 1)
                                    .concat(futureTimeFeat, 1)
                            : pastTimeFeat.get(":, {}:", -contextLength + 1);

            NDArray features = expandedStaticFeat.concat(timeFeat, -1);
            NDArray lags = laggedSequenceValues(lagsSeq, priorSequence, sequence);

            NDArray rnnInput = lags.concat(features, -1);

            NDList outputs = rnn.forward(ps, new NDList(rnnInput), training);
            NDArray output = outputs.get(0);
            NDArray hiddenState = outputs.get(1);
            NDArray cellState = outputs.get(2);

            NDList params = paramProj.forward(ps, new NDList(output), training);

            scale.setName("scale");
            output.setName("output");
            staticFeat.setName("static_feat");
            hiddenState.setName("hidden_state");
            cellState.setName("cell_state");
            return scope.ret(
                    params.addAll(new NDList(scale, output, staticFeat, hiddenState, cellState)));
        }
    }

    /**
     * Construct an {@link NDArray} of lagged values from a given sequence.
     *
     * @param indices indices of lagged observations
     * @param priorSequence the input sequence prior to the time range for which the output is
     *     required
     * @param sequence the input sequence in the time range where the output is required
     * @return the lagged values
     */
    protected NDArray laggedSequenceValues(
            List<Integer> indices, NDArray priorSequence, NDArray sequence) {
        if (Collections.max(indices) > (int) priorSequence.getShape().get(1)) {
            throw new IllegalArgumentException(
                    String.format(
                            "lags cannot go further than prior sequence length, found lag %d while"
                                    + " prior sequence is only %d-long",
                            Collections.max(indices), priorSequence.getShape().get(1)));
        }
        try (NDManager scope = NDManager.subManagerOf(priorSequence)) {
            scope.tempAttachAll(priorSequence, sequence);
            NDArray fullSequence = priorSequence.concat(sequence, 1);

            NDList lagsValues = new NDList(indices.size());
            for (int lagIndex : indices) {
                long begin = -lagIndex - sequence.getShape().get(1);
                long end = -lagIndex;
                lagsValues.add(
                        end < 0
                                ? fullSequence.get(":, {}:{}", begin, end)
                                : fullSequence.get(":, {}:", begin));
            }

            NDArray lags = NDArrays.stack(lagsValues, -1);
            return scope.ret(lags.reshape(lags.getShape().get(0), lags.getShape().get(1), -1));
        }
    }

    /**
     * Return the context length.
     *
     * @return the context length
     */
    public int getContextLength() {
        return contextLength;
    }

    /**
     * Return the history length.
     *
     * @return the history length
     */
    public int getHistoryLength() {
        return historyLength;
    }

    /**
     * Construct a training transformation of deepar model.
     *
     * @param manager the {@link NDManager} to create value
     * @return the transformation
     */
    public List<TimeSeriesTransform> createTrainingTransformation(NDManager manager) {
        List<TimeSeriesTransform> transformation = createTransformation(manager);

        InstanceSampler sampler = new ExpectedNumInstanceSampler(0, 0, predictionLength, 1.0);
        transformation.add(
                new InstanceSplit(
                        FieldName.TARGET,
                        FieldName.IS_PAD,
                        FieldName.START,
                        FieldName.FORECAST_START,
                        sampler,
                        historyLength,
                        predictionLength,
                        new FieldName[] {FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES},
                        distrOutput.getValueInSupport()));

        transformation.add(new SelectField(TRAIN_INPUT_FIELDS));
        return transformation;
    }

    /**
     * Construct a prediction transformation of deepar model.
     *
     * @param manager the {@link NDManager} to create value
     * @return the transformation
     */
    public List<TimeSeriesTransform> createPredictionTransformation(NDManager manager) {
        List<TimeSeriesTransform> transformation = createTransformation(manager);

        InstanceSampler sampler = PredictionSplitSampler.newValidationSplitSampler();
        transformation.add(
                new InstanceSplit(
                        FieldName.TARGET,
                        FieldName.IS_PAD,
                        FieldName.START,
                        FieldName.FORECAST_START,
                        sampler,
                        historyLength,
                        predictionLength,
                        new FieldName[] {FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES},
                        distrOutput.getValueInSupport()));

        transformation.add(new SelectField(PRED_INPUT_FIELDS));
        return transformation;
    }

    private List<TimeSeriesTransform> createTransformation(NDManager manager) {
        List<TimeSeriesTransform> transformation = new ArrayList<>();

        List<FieldName> removeFieldNames = new ArrayList<>();
        removeFieldNames.add(FieldName.FEAT_DYNAMIC_CAT);
        if (!useFeatStaticReal) {
            removeFieldNames.add(FieldName.FEAT_STATIC_REAL);
        }
        if (!useFeatDynamicReal) {
            removeFieldNames.add(FieldName.FEAT_DYNAMIC_REAL);
        }

        transformation.add(new RemoveFields(removeFieldNames));
        if (!useFeatStaticCat) {
            transformation.add(
                    new SetField(FieldName.FEAT_STATIC_CAT, manager.zeros(new Shape(1))));
        }
        if (!useFeatDynamicReal) {
            transformation.add(
                    new SetField(FieldName.FEAT_STATIC_REAL, manager.zeros(new Shape(1))));
        }

        transformation.add(new AsArray(FieldName.FEAT_STATIC_CAT, 1, DataType.INT32));
        transformation.add(new AsArray(FieldName.FEAT_STATIC_REAL, 1));

        transformation.add(
                new AddObservedValuesIndicator(FieldName.TARGET, FieldName.OBSERVED_VALUES));

        transformation.add(
                new AddTimeFeature(
                        FieldName.START,
                        FieldName.TARGET,
                        FieldName.FEAT_TIME,
                        TimeFeature.timeFeaturesFromFreqStr(freq),
                        predictionLength,
                        freq));

        transformation.add(
                new AddAgeFeature(FieldName.TARGET, FieldName.FEAT_AGE, predictionLength, true));

        FieldName[] inputFields;
        if (!useFeatDynamicReal) {
            inputFields = new FieldName[] {FieldName.FEAT_TIME, FieldName.FEAT_AGE};
        } else {
            inputFields =
                    new FieldName[] {
                        FieldName.FEAT_TIME, FieldName.FEAT_AGE, FieldName.FEAT_DYNAMIC_REAL
                    };
        }
        transformation.add(new VstackFeatures(FieldName.FEAT_TIME, inputFields));

        return transformation;
    }

    /**
     * Create a builder to build a {@code DeepARTrainingNetwork} or {@code DeepARPredictionNetwork}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * The builder to construct a {@code DeepARTrainingNetwork} or {@code DeepARPredictionNetwork}.
     * type of {@link ai.djl.nn.Block}.
     */
    public static final class Builder {

        private String freq;
        private int contextLength;
        private int predictionLength;
        private int numParallelSamples = 100;
        private int numLayers = 2;
        private int hiddenSize = 40;
        private float dropRate = 0.1f;

        private boolean useFeatDynamicReal;
        private boolean useFeatStaticCat;
        private boolean useFeatStaticReal;
        private boolean scaling = true;

        private DistributionOutput distrOutput = new StudentTOutput();
        private List<Integer> cardinality;
        private List<Integer> embeddingDimension;
        private List<Integer> lagsSeq;

        /**
         * Set the prediction frequency.
         *
         * @param freq the frequency
         * @return this builder
         */
        public Builder setFreq(String freq) {
            this.freq = freq;
            return this;
        }

        /**
         * Set the prediction length.
         *
         * @param predictionLength the prediction length
         * @return this builder
         */
        public Builder setPredictionLength(int predictionLength) {
            this.predictionLength = predictionLength;
            return this;
        }

        /**
         * Set the cardinality for static categorical feature.
         *
         * @param cardinality the cardinality
         * @return this builder
         */
        public Builder setCardinality(List<Integer> cardinality) {
            this.cardinality = cardinality;
            return this;
        }

        /**
         * Set the optional {@link DistributionOutput} default {@link StudentTOutput}.
         *
         * @param distrOutput the {@link DistributionOutput}
         * @return this builder
         */
        public Builder optDistrOutput(DistributionOutput distrOutput) {
            this.distrOutput = distrOutput;
            return this;
        }

        /**
         * Set the optional context length.
         *
         * @param contextLength the context length
         * @return this builder
         */
        public Builder optContextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        /**
         * Set the optional number parallel samples.
         *
         * @param numParallelSamples the num parallel samples
         * @return this builder
         */
        public Builder optNumParallelSamples(int numParallelSamples) {
            this.numParallelSamples = numParallelSamples;
            return this;
        }

        /**
         * Set the optional number of rnn layers.
         *
         * @param numLayers the number of rnn layers
         * @return this builder
         */
        public Builder optNumLayers(int numLayers) {
            this.numLayers = numLayers;
            return this;
        }

        /**
         * Set the optional number of rnn hidden size.
         *
         * @param hiddenSize the number of rnn hidden size
         * @return this builder
         */
        public Builder optHiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        /**
         * Set the optional number of rnn drop rate.
         *
         * @param dropRate the number of rnn drop rate
         * @return this builder
         */
        public Builder optDropRate(float dropRate) {
            this.dropRate = dropRate;
            return this;
        }

        /**
         * Set the optional embedding dimension.
         *
         * @param embeddingDimension the embedding dimension
         * @return this builder
         */
        public Builder optEmbeddingDimension(List<Integer> embeddingDimension) {
            this.embeddingDimension = embeddingDimension;
            return this;
        }

        /**
         * Set the optional lags sequence, default generate from frequency.
         *
         * @param lagsSeq the lags sequence
         * @return this builder
         */
        public Builder optLagsSeq(List<Integer> lagsSeq) {
            this.lagsSeq = lagsSeq;
            return this;
        }

        /**
         * Set whether to use dynamic real feature.
         *
         * @param useFeatDynamicReal whether to use dynamic real feature
         * @return this builder
         */
        public Builder optUseFeatDynamicReal(boolean useFeatDynamicReal) {
            this.useFeatDynamicReal = useFeatDynamicReal;
            return this;
        }

        /**
         * Set whether to use static categorical feature.
         *
         * @param useFeatStaticCat whether to use static categorical feature
         * @return this builder
         */
        public Builder optUseFeatStaticCat(boolean useFeatStaticCat) {
            this.useFeatStaticCat = useFeatStaticCat;
            return this;
        }

        /**
         * Set whether to use static real feature.
         *
         * @param useFeatStaticReal whether to use static real feature
         * @return this builder
         */
        public Builder optUseFeatStaticReal(boolean useFeatStaticReal) {
            this.useFeatStaticReal = useFeatStaticReal;
            return this;
        }

        /**
         * Build a {@link DeepARTrainingNetwork} block.
         *
         * @return the {@link DeepARTrainingNetwork} block.
         */
        public DeepARTrainingNetwork buildTrainingNetwork() {
            return new DeepARTrainingNetwork(this);
        }

        /**
         * Build a {@link DeepARPredictionNetwork} block.
         *
         * @return the {@link DeepARPredictionNetwork} block.
         */
        public DeepARPredictionNetwork buildPredictionNetwork() {
            return new DeepARPredictionNetwork(this);
        }
    }
}
