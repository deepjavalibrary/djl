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
import ai.djl.basicdataset.StanfordMovieReview;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.ModelZooTextEmbedding;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.modality.nlp.preprocess.TextTruncator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DataManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.commons.cli.ParseException;

public final class TrainSentimentAnalysis {
    private static final int PADDING_SIZE = 500;
    private static final List<TextProcessor> TEXT_PROCESSORS =
            Arrays.asList(
                    new SimpleTokenizer(),
                    new LowerCaseConvertor(Locale.ENGLISH),
                    new PunctuationSeparator(),
                    new TextTruncator(PADDING_SIZE));
    private static int paddingTokenValue;

    private TrainSentimentAnalysis() {}

    public static void main(String[] args)
            throws IOException, ParseException, ModelNotFoundException, MalformedModelException,
                    TranslateException {
        TrainSentimentAnalysis.runExample(args);
    }

    public static TrainingResult runExample(String[] args)
            throws IOException, ParseException, ModelNotFoundException, MalformedModelException,
                    TranslateException {
        Arguments arguments = Arguments.parseArgs(args);
        ExecutorService executorService = Executors.newFixedThreadPool(8);
        Criteria<String, NDList> criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.WORD_EMBEDDING)
                        .setTypes(String.class, NDList.class)
                        .optArtifactId("glove")
                        .optFilter("dimensions", "50")
                        .build();

        try (Model model = Model.newInstance();
                ZooModel<String, NDList> embedding = ModelZoo.loadModel(criteria)) {
            ModelZooTextEmbedding modelZooTextEmbedding = new ModelZooTextEmbedding(embedding);
            // get training and validation dataset
            paddingTokenValue =
                    modelZooTextEmbedding
                            .preprocessTextToEmbed(Collections.singletonList("<unk>"))[0];
            StanfordMovieReview trainingSet =
                    getDataset(embedding, Dataset.Usage.TRAIN, executorService, arguments);
            StanfordMovieReview validateSet =
                    getDataset(embedding, Dataset.Usage.TEST, executorService, arguments);
            model.setBlock(getModel());

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments, modelZooTextEmbedding);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());
                Shape encoderInputShape = new Shape(arguments.getBatchSize(), 10, 50);

                // initialize trainer with proper input shape
                trainer.initialize(encoderInputShape);

                TrainingUtils.fit(
                        trainer,
                        arguments.getEpoch(),
                        trainingSet,
                        validateSet,
                        null,
                        "stanfordSentimentAnalysis");

                TrainingResult result = trainer.getTrainingResult();
                model.setProperty("Epoch", String.valueOf(result.getEpoch()));
                model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

                Path modelSavePath = Paths.get(arguments.getOutputDir());
                model.save(modelSavePath, "stanfordSentimentAnalysis");

                try (Predictor<String, Boolean> predictor =
                        model.newPredictor(new Translator(embedding))) {
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

    private static Block getModel() {
        return new SequentialBlock()
                .add(
                        LSTM.builder()
                                .setNumStackedLayers(2)
                                .setStateSize(100)
                                .setSequenceLength(false)
                                .optBidrectional(true)
                                .build())
                .add(
                        new LambdaBlock(
                                x -> {
                                    long sequenceLength = x.head().getShape().get(1);
                                    NDArray ntc = x.head().transpose(1, 0, 2);
                                    return new NDList(
                                            NDArrays.concat(
                                                    new NDList(
                                                            ntc.get(0),
                                                            ntc.get(sequenceLength - 1)),
                                                    1));
                                }))
                .add(Linear.builder().setOutChannels(2).build());
    }

    public static DefaultTrainingConfig setupTrainingConfig(
            Arguments arguments, ModelZooTextEmbedding embedding) {
        return new DefaultTrainingConfig(new SoftmaxCrossEntropyLoss())
                .optDataManager(new EmbeddingDataManager(embedding))
                .optDevices(Device.getDevices(arguments.getMaxGpus()))
                .addTrainingListeners(TrainingListener.Defaults.logging(arguments.getOutputDir()));
    }

    public static StanfordMovieReview getDataset(
            Model embeddingModel,
            Dataset.Usage usage,
            ExecutorService executorService,
            Arguments arguments)
            throws IOException {
        StanfordMovieReview stanfordMovieReview =
                StanfordMovieReview.builder()
                        .setSampling(arguments.getBatchSize(), true, false)
                        .optDataBatchifier(
                                PaddingStackBatchifier.builder()
                                        .optIncludeValidLengths(false)
                                        .addPad(
                                                0,
                                                0,
                                                (m) -> m.ones(new Shape(1)).mul(paddingTokenValue),
                                                PADDING_SIZE)
                                        .build())
                        .setSourceConfiguration(
                                new TextData.Configuration()
                                        .setTextEmbedding(new ModelZooTextEmbedding(embeddingModel))
                                        .setTextProcessors(TEXT_PROCESSORS))
                        .setUsage(usage)
                        .optExecutor(executorService, 8)
                        .optLimit(arguments.getLimit())
                        .build();
        stanfordMovieReview.prepare(new ProgressBar());
        return stanfordMovieReview;
    }

    public static class Translator implements ai.djl.translate.Translator<String, Boolean> {
        private TextEmbedding textEmbedding;
        private NDManager manager;

        public Translator(ZooModel<String, NDList> embeddingModel) {
            textEmbedding = new ModelZooTextEmbedding(embeddingModel);
            manager = embeddingModel.getNDManager();
        }

        @Override
        public Boolean processOutput(TranslatorContext ctx, NDList list) throws Exception {
            long argmax = list.head().argMax().getLong();
            return argmax == 1;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) throws EmbeddingException {
            List<String> tokens = Collections.singletonList(input);
            for (TextProcessor processor : TEXT_PROCESSORS) {
                tokens = processor.preprocess(tokens);
            }
            NDArray array = textEmbedding.embedText(manager, tokens);
            return new NDList(array);
        }

        /** {@inheritDoc} */
        @Override
        public Batchifier getBatchifier() {
            return PaddingStackBatchifier.builder()
                    .optIncludeValidLengths(false)
                    .addPad(0, 0, m -> m.ones(new Shape(1, 50)).mul(paddingTokenValue))
                    .build();
        }
    }

    private static class EmbeddingDataManager extends DataManager {
        private ModelZooTextEmbedding embedding;

        public EmbeddingDataManager(ModelZooTextEmbedding embedding) {
            this.embedding = embedding;
        }

        @Override
        public NDList getData(Batch batch) {
            try {
                return new NDList(embedding.embedText(batch.getData().head()));
            } catch (EmbeddingException e) {
                throw new IllegalArgumentException(e.getMessage(), e);
            }
        }
    }
}
