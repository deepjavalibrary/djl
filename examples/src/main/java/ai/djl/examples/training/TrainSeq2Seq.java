/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicdataset.TatoebaEnglishFrenchDataset;
import ai.djl.basicdataset.TextDataset;
import ai.djl.basicmodelzoo.nlp.SimpleSequenceDecoder;
import ai.djl.basicmodelzoo.nlp.SimpleSequenceEncoder;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.metric.Metrics;
import ai.djl.modality.nlp.EncoderDecoder;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.DataManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.MaskedSoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.PaddingStackBatchifier;
import java.io.IOException;
import java.nio.file.Paths;
import org.apache.commons.cli.ParseException;

public final class TrainSeq2Seq {
    private TrainSeq2Seq() {}

    public static void main(String[] args) throws IOException, ParseException {
        TrainSeq2Seq.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, ParseException {
        Arguments arguments = Arguments.parseArgs(args);

        try (Model model = Model.newInstance()) {
            // get training and validation dataset
            TextDataset trainingSet =
                    getDataset(Dataset.Usage.TRAIN, arguments, model.getNDManager());
            TextDataset validateSet =
                    getDataset(Dataset.Usage.TEST, arguments, model.getNDManager());

            model.setBlock(getSeq2SeqModel(trainingSet));

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments);
            config.addTrainingListeners(
                    TrainingListener.Defaults.logging(arguments.getOutputDir()));

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                /*
                In Sequence-Sequence model for MT, the decoder input must be staggered by one wrt
                the label during training.
                 */
                Shape encoderInputShape = new Shape(arguments.getBatchSize(), 10);
                Shape decoderInputShape = new Shape(arguments.getBatchSize(), 9);

                // initialize trainer with proper input shape
                trainer.initialize(encoderInputShape, decoderInputShape);

                TrainingUtils.fit(
                        trainer,
                        arguments.getEpoch(),
                        trainingSet,
                        validateSet,
                        arguments.getOutputDir(),
                        "seq2seqEn2Fr");

                TrainingResult result = trainer.getTrainingResult();
                model.setProperty("Epoch", String.valueOf(result.getEpoch()));
                model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

                model.save(Paths.get(arguments.getOutputDir()), "seq2seqEn2Fr");
                return result;
            }
        }
    }

    private static Block getSeq2SeqModel(TextDataset dataset) {
        SimpleSequenceEncoder simpleSequenceEncoder =
                new SimpleSequenceEncoder(
                        (TrainableTextEmbedding) dataset.getTextEmbedding(true),
                        new LSTM.Builder()
                                .setStateSize(32)
                                .setNumStackedLayers(2)
                                .optDropRate(0)
                                .build());
        SimpleSequenceDecoder simpleSequenceDecoder =
                new SimpleSequenceDecoder(
                        (TrainableTextEmbedding) dataset.getTextEmbedding(false),
                        new LSTM.Builder()
                                .setStateSize(32)
                                .setNumStackedLayers(2)
                                .optDropRate(0)
                                .build(),
                        dataset.getVocabulary(false).size());
        return new EncoderDecoder(simpleSequenceEncoder, simpleSequenceDecoder);
    }

    public static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        return new DefaultTrainingConfig(new MaskedSoftmaxCrossEntropyLoss())
                .optInitializer(new XavierInitializer())
                .optOptimizer(
                        Adam.builder()
                                .optLearningRateTracker(
                                        LearningRateTracker.fixedLearningRate(0.005f))
                                .build())
                .optDevices(Device.getDevices(arguments.getMaxGpus()))
                .optDataManager(new Seq2SeqDataManager());
    }

    public static TextDataset getDataset(
            Dataset.Usage usage, Arguments arguments, NDManager manager) throws IOException {
        TatoebaEnglishFrenchDataset tatoebaEnglishFrenchDataset =
                TatoebaEnglishFrenchDataset.builder()
                        .setSampling(arguments.getBatchSize(), true, true)
                        .optBatchier(
                                PaddingStackBatchifier.builder()
                                        .optIncludeValidLengths(true)
                                        .addPad(0, 0, (m) -> m.zeros(new Shape(1)))
                                        .build())
                        .optEmbeddingSize(32)
                        .optUsage(usage)
                        .optLimit(arguments.getLimit())
                        .build();
        tatoebaEnglishFrenchDataset.prepare(new ProgressBar());
        return tatoebaEnglishFrenchDataset;
    }

    private static class Seq2SeqDataManager extends DataManager {
        @Override
        public NDList getData(Batch batch) {
            NDList data = new NDList();
            data.add(batch.getData().head());
            NDArray target = batch.getLabels().head().get(new NDIndex(":, :-1"));
            data.add(target);
            return data;
        }

        @Override
        public NDList getLabels(Batch batch) {
            NDList labels = batch.getLabels();
            return new NDList(labels.head().get(new NDIndex(":, 1:")), labels.get(1));
        }
    }
}
