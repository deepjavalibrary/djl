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

import ai.djl.Model;
import ai.djl.basicdataset.nlp.TatoebaEnglishFrenchDataset;
import ai.djl.basicdataset.nlp.TextDataset;
import ai.djl.basicdataset.utils.TextData.Configuration;
import ai.djl.basicmodelzoo.nlp.SimpleTextDecoder;
import ai.djl.basicmodelzoo.nlp.SimpleTextEncoder;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.metric.Metrics;
import ai.djl.modality.nlp.EncoderDecoder;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextTerminator;
import ai.djl.modality.nlp.preprocess.TextTruncator;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.MaskedSoftmaxCrossEntropyLoss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class TrainSeq2Seq {

    private TrainSeq2Seq() {}

    public static void main(String[] args) throws IOException, TranslateException {
        TrainSeq2Seq.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        ExecutorService executorService = Executors.newFixedThreadPool(8);
        try (Model model = Model.newInstance("seq2seqMTEn-Fr")) {
            // get training and validation dataset
            TextDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments, null, null);
            // Fetch TextEmbedding from dataset
            TrainableTextEmbedding sourceEmbedding =
                    (TrainableTextEmbedding) trainingSet.getTextEmbedding(true);
            TrainableTextEmbedding targetEmbedding =
                    (TrainableTextEmbedding) trainingSet.getTextEmbedding(false);

            // Build the model with the TextEmbedding so that embeddings can be trained
            Block block =
                    getSeq2SeqModel(
                            sourceEmbedding,
                            targetEmbedding,
                            trainingSet.getVocabulary(false).size());
            model.setBlock(block);

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments, executorService);

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

                // EncoderDecoder don't implement inference, set validateDataset to null
                EasyTrain.fit(trainer, arguments.getEpoch(), trainingSet, null);

                return trainer.getTrainingResult();
            } finally {
                executorService.shutdownNow();
            }
        }
    }

    private static Block getSeq2SeqModel(
            TrainableTextEmbedding sourceEmbedding,
            TrainableTextEmbedding targetEmbedding,
            long vocabSize) {
        SimpleTextEncoder simpleTextEncoder =
                new SimpleTextEncoder(
                        sourceEmbedding,
                        new LSTM.Builder()
                                .setStateSize(32)
                                .setNumLayers(2)
                                .optDropRate(0)
                                .optBatchFirst(true)
                                .optReturnState(true)
                                .build());
        SimpleTextDecoder simpleTextDecoder =
                new SimpleTextDecoder(
                        targetEmbedding,
                        new LSTM.Builder()
                                .setStateSize(32)
                                .setNumLayers(2)
                                .optDropRate(0)
                                .optBatchFirst(true)
                                .optReturnState(false)
                                .build(),
                        vocabSize);
        return new EncoderDecoder(simpleTextEncoder, simpleTextDecoder);
    }

    public static DefaultTrainingConfig setupTrainingConfig(
            Arguments arguments, ExecutorService executorService) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(new MaskedSoftmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy("Accuracy", 0, 2))
                .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                .optExecutorService(executorService)
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    public static TextDataset getDataset(
            Dataset.Usage usage,
            Arguments arguments,
            TextEmbedding sourceEmbedding,
            TextEmbedding targetEmbedding)
            throws IOException, TranslateException {
        long limit =
                usage == Dataset.Usage.TRAIN ? arguments.getLimit() : arguments.getLimit() / 10;
        TatoebaEnglishFrenchDataset.Builder datasetBuilder =
                TatoebaEnglishFrenchDataset.builder()
                        .setSampling(arguments.getBatchSize(), true, false)
                        .optDataBatchifier(
                                PaddingStackBatchifier.builder()
                                        .optIncludeValidLengths(true)
                                        .addPad(0, 0, (m) -> m.zeros(new Shape(1)), 10)
                                        .build())
                        .optLabelBatchifier(
                                PaddingStackBatchifier.builder()
                                        .optIncludeValidLengths(true)
                                        .addPad(0, 0, (m) -> m.ones(new Shape(1)), 10)
                                        .build())
                        .optUsage(usage)
                        .optPrefetchNumber(8)
                        .optLimit(limit);
        Configuration sourceConfig =
                new Configuration()
                        .setTextProcessors(
                                Arrays.asList(
                                        new SimpleTokenizer(),
                                        new LowerCaseConvertor(Locale.ENGLISH),
                                        new PunctuationSeparator(),
                                        new TextTruncator(10)));
        Configuration targetConfig =
                new Configuration()
                        .setTextProcessors(
                                Arrays.asList(
                                        new SimpleTokenizer(),
                                        new LowerCaseConvertor(Locale.FRENCH),
                                        new PunctuationSeparator(),
                                        new TextTruncator(8),
                                        new TextTerminator()));
        if (sourceEmbedding != null) {
            sourceConfig.setTextEmbedding(sourceEmbedding);
        } else {
            sourceConfig.setEmbeddingSize(32);
        }
        if (targetEmbedding != null) {
            targetConfig.setTextEmbedding(targetEmbedding);
        } else {
            targetConfig.setEmbeddingSize(32);
        }
        TatoebaEnglishFrenchDataset dataset =
                datasetBuilder
                        .setSourceConfiguration(sourceConfig)
                        .setTargetConfiguration(targetConfig)
                        .build();
        dataset.prepare(new ProgressBar());
        return dataset;
    }
}
