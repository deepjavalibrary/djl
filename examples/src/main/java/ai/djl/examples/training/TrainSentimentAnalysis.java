/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.nlp.StanfordMovieReview;
import ai.djl.basicdataset.utils.FixedBucketSampler;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.examples.training.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.ModelZooTextEmbedding;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class TrainSentimentAnalysis {
    private static final List<TextProcessor> TEXT_PROCESSORS =
            Arrays.asList(
                    new SimpleTokenizer(),
                    new LowerCaseConvertor(Locale.ENGLISH),
                    new PunctuationSeparator());
    private static long paddingTokenValue;

    private TrainSentimentAnalysis() {}

    public static void main(String[] args)
            throws IOException, ModelNotFoundException, MalformedModelException,
                    TranslateException {
        TrainSentimentAnalysis.runExample(args);
    }

    public static TrainingResult runExample(String[] args)
            throws IOException, ModelNotFoundException, MalformedModelException,
                    TranslateException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        ExecutorService executorService = Executors.newFixedThreadPool(8);
        Criteria<String, NDList> criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.WORD_EMBEDDING)
                        .setTypes(String.class, NDList.class)
                        .optArtifactId("glove")
                        .optFilter("dimensions", "50")
                        .build();

        try (Model model = Model.newInstance("stanfordSentimentAnalysis");
                ZooModel<String, NDList> embedding = criteria.loadModel()) {
            ModelZooTextEmbedding modelZooTextEmbedding = new ModelZooTextEmbedding(embedding);
            // get training and validation dataset
            paddingTokenValue =
                    modelZooTextEmbedding
                            .preprocessTextToEmbed(Collections.singletonList("<unk>"))[0];
            StanfordMovieReview trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);
            StanfordMovieReview validateSet = getDataset(Dataset.Usage.TEST, arguments);
            model.setBlock(getModel(modelZooTextEmbedding));

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments, executorService);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());
                Shape encoderInputShape = new Shape(arguments.getBatchSize(), 10);

                // initialize trainer with proper input shape
                trainer.initialize(encoderInputShape);

                EasyTrain.fit(trainer, arguments.getEpoch(), trainingSet, validateSet);

                TrainingResult result = trainer.getTrainingResult();

                try (Predictor<String, Boolean> predictor =
                        model.newPredictor(new MyTranslator(embedding))) {
                    List<String> sentences =
                            Arrays.asList(
                                    "This movie was very good",
                                    "This movie was terrible",
                                    "The movie was not that great");
                    System.out.println(predictor.batchPredict(sentences)); // NOPMD
                }

                return result;
            }
        } finally {
            executorService.shutdownNow();
        }
    }

    private static Block getModel(ModelZooTextEmbedding embedding) {
        return new SequentialBlock()
                .addSingleton(
                        a -> {
                            try {
                                return embedding.embedText(a);
                            } catch (EmbeddingException e) {
                                throw new IllegalStateException(e);
                            }
                        })
                .add(
                        LSTM.builder()
                                .setNumLayers(2)
                                .setStateSize(100)
                                .optBidirectional(true)
                                .build())
                .add(
                        x -> {
                            long sequenceLength = x.head().getShape().get(1);
                            NDArray ntc = x.head().transpose(1, 0, 2);
                            return new NDList(
                                    NDArrays.concat(
                                            new NDList(ntc.get(0), ntc.get(sequenceLength - 1)),
                                            1));
                        })
                .add(Linear.builder().setUnits(2).build());
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

        return new DefaultTrainingConfig(new SoftmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optDevices(Device.getDevices(arguments.getMaxGpus()))
                .optExecutorService(executorService)
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    public static StanfordMovieReview getDataset(Dataset.Usage usage, Arguments arguments)
            throws IOException, TranslateException {
        StanfordMovieReview stanfordMovieReview =
                StanfordMovieReview.builder()
                        .setSampling(new FixedBucketSampler(arguments.getBatchSize()))
                        .optDataBatchifier(
                                PaddingStackBatchifier.builder()
                                        .optIncludeValidLengths(false)
                                        .addPad(
                                                0,
                                                0,
                                                (m) -> m.ones(new Shape(1)).mul(paddingTokenValue))
                                        .build())
                        .setSourceConfiguration(
                                new TextData.Configuration().setTextProcessors(TEXT_PROCESSORS))
                        .optUsage(usage)
                        .optPrefetchNumber(8)
                        .optLimit(arguments.getLimit())
                        .build();
        stanfordMovieReview.prepare(new ProgressBar());
        return stanfordMovieReview;
    }

    public static final class MyTranslator implements Translator<String, Boolean> {

        private TextEmbedding textEmbedding;
        private NDManager manager;

        public MyTranslator(ZooModel<String, NDList> embeddingModel) {
            textEmbedding = new ModelZooTextEmbedding(embeddingModel);
            manager = embeddingModel.getNDManager();
        }

        @Override
        public Boolean processOutput(TranslatorContext ctx, NDList list) {
            long argmax = list.head().argMax().getLong();
            return argmax == 1;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            List<String> tokens = Collections.singletonList(input);
            for (TextProcessor processor : TEXT_PROCESSORS) {
                tokens = processor.preprocess(tokens);
            }
            NDArray array = manager.create(textEmbedding.preprocessTextToEmbed(tokens));
            return new NDList(array);
        }

        /** {@inheritDoc} */
        @Override
        public Batchifier getBatchifier() {
            return PaddingStackBatchifier.builder()
                    .optIncludeValidLengths(false)
                    .addPad(0, 0, m -> m.ones(new Shape(1)).mul(paddingTokenValue))
                    .build();
        }
    }
}
