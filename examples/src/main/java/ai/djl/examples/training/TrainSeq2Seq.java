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
import ai.djl.basicdataset.utils.TextData.Configuration;
import ai.djl.basicmodelzoo.nlp.SimpleSequenceDecoder;
import ai.djl.basicmodelzoo.nlp.SimpleSequenceEncoder;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.metric.Metrics;
import ai.djl.modality.nlp.EncoderDecoder;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextTerminator;
import ai.djl.modality.nlp.preprocess.TextTruncator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
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
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.MaskedSoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.PaddingStackBatchifier;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.commons.cli.ParseException;

public final class TrainSeq2Seq {
    private TrainSeq2Seq() {}

    public static void main(String[] args) throws IOException, ParseException {
        TrainSeq2Seq.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, ParseException {
        Arguments arguments = Arguments.parseArgs(args);
        ExecutorService executorService = Executors.newFixedThreadPool(8);
        try (Model model = Model.newInstance()) {
            // get training and validation dataset
            TextDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments, executorService);
            TextDataset validateDataset =
                    getDataset(Dataset.Usage.TEST, arguments, executorService);

            // Fetch TextEmbedding from dataset
            TrainableTextEmbedding sourceTextEmbedding =
                    (TrainableTextEmbedding) trainingSet.getTextEmbedding(true);
            TrainableTextEmbedding targetTextEmbedding =
                    (TrainableTextEmbedding) trainingSet.getTextEmbedding(false);

            // Build the model with the TextEmbedding so that embeddings can be trained
            Block block =
                    getSeq2SeqModel(
                            sourceTextEmbedding,
                            targetTextEmbedding,
                            trainingSet.getVocabulary(false).getAllTokens().size());
            model.setBlock(block);

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
                        validateDataset,
                        arguments.getOutputDir(),
                        "seq2seqMTEn-Fr");
                TrainingResult result = trainer.getTrainingResult();
                model.setProperty("Epoch", String.valueOf(result.getEpoch()));
                model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                model.save(Paths.get(arguments.getOutputDir()), "seq2seqMTEn-Fr");
                return result;
            } finally {
                executorService.shutdownNow();
            }
        }
    }

    private static Block getSeq2SeqModel(
            TrainableTextEmbedding sourceEmbedding,
            TrainableTextEmbedding targetEmbedding,
            int vocabSize) {
        SimpleSequenceEncoder simpleSequenceEncoder =
                new SimpleSequenceEncoder(
                        sourceEmbedding,
                        new LSTM.Builder()
                                .setStateSize(32)
                                .setNumStackedLayers(2)
                                .optDropRate(0)
                                .build());
        SimpleSequenceDecoder simpleSequenceDecoder =
                new SimpleSequenceDecoder(
                        targetEmbedding,
                        new LSTM.Builder()
                                .setStateSize(32)
                                .setNumStackedLayers(2)
                                .optDropRate(0)
                                .build(),
                        vocabSize);
        return new EncoderDecoder(simpleSequenceEncoder, simpleSequenceDecoder);
    }

    public static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        return new DefaultTrainingConfig(new MaskedSoftmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy("Accuracy", 0, 2))
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
            Dataset.Usage usage, Arguments arguments, ExecutorService executorService)
            throws IOException {
        long limit =
                usage == Dataset.Usage.TRAIN ? arguments.getLimit() : arguments.getLimit() / 10;
        TatoebaEnglishFrenchDataset tatoebaEnglishFrenchDataset =
                TatoebaEnglishFrenchDataset.builder()
                        .setSampling(arguments.getBatchSize(), true, false)
                        .optBatchier(
                                PaddingStackBatchifier.builder()
                                        .optIncludeValidLengths(true)
                                        .addPad(0, 0, (m) -> m.zeros(new Shape(1)), 10)
                                        .build())
                        .setSourceConfiguration(
                                new Configuration().setEmbeddingSize(32).setTrainEmbedding(true))
                        .setTargetConfiguration(
                                new Configuration()
                                        .setEmbeddingSize(32)
                                        .setTrainEmbedding(true)
                                        .setTextProcessors(
                                                Arrays.asList(
                                                        new SimpleTokenizer(),
                                                        new LowerCaseConvertor(Locale.FRENCH),
                                                        new PunctuationSeparator(),
                                                        new TextTruncator(8),
                                                        new TextTerminator())))
                        .optUsage(usage)
                        .optExecutor(executorService, 8)
                        .optLimit(limit)
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
            return new NDList(labels.head().get(new NDIndex(":, 1:")), labels.get(1).sub(1));
        }
    }
}
