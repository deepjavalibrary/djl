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
package ai.djl.examples.training.transferlearning;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.tabular.CsvDataset;
import ai.djl.basicdataset.tabular.utils.DynamicBuffer;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.basicdataset.tabular.utils.Featurizer.DataFeaturizer;
import ai.djl.examples.training.util.Arguments;
import ai.djl.metric.Metrics;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.TranslateException;

import org.apache.commons.csv.CSVFormat;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Locale;

public final class TrainAmazonReviewRanking {

    private TrainAmazonReviewRanking() {}

    public static void main(String[] args)
            throws IOException, TranslateException, ModelException, URISyntaxException {
        TrainAmazonReviewRanking.runExample(args);
    }

    public static TrainingResult runExample(String[] args)
            throws IOException, TranslateException, ModelException, URISyntaxException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        String engine = arguments.getEngine();
        String modelUrls =
                "https://resources.djl.ai/test-models/traced_distilbert_wikipedia_uncased.zip";

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.WORD_EMBEDDING)
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(modelUrls)
                        .optEngine(engine)
                        .optProgress(new ProgressBar())
                        .optOption("trainParam", "true")
                        .build();
        int maxTokenLength = 64;
        try (Model model = Model.newInstance("AmazonReviewRatingClassification", engine);
                ZooModel<NDList, NDList> embedding = criteria.loadModel()) {
            // Prepare the vocabulary
            DefaultVocabulary vocabulary =
                    DefaultVocabulary.builder()
                            .addFromTextFile(embedding.getArtifact("vocab.txt"))
                            .optUnknownToken("[UNK]")
                            .build();
            // Prepare dataset
            BertFullTokenizer tokenizer = new BertFullTokenizer(vocabulary, true);
            CsvDataset amazonReviewDataset = getDataset(arguments, tokenizer, maxTokenLength);
            // split data with 7:3 train:valid ratio
            RandomAccessDataset[] datasets = amazonReviewDataset.randomSplit(7, 3);
            RandomAccessDataset trainingSet = datasets[0];
            RandomAccessDataset validationSet = datasets[1];
            // create training model
            model.setBlock(getBlock(embedding.getBlock()));
            DefaultTrainingConfig config = setupTrainingConfig(arguments);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());
                Shape encoderInputShape = new Shape(arguments.getBatchSize(), maxTokenLength);
                // initialize trainer with proper input shape
                trainer.initialize(encoderInputShape);

                EasyTrain.fit(trainer, arguments.getEpoch(), trainingSet, validationSet);
                return trainer.getTrainingResult();
            }
            // model.save("your-model-path"); // save the model
        }
    }

    private static CsvDataset getDataset(
            Arguments arguments, BertFullTokenizer tokenizer, int maxLength) {
        String amazonReview =
                "https://mlrepo.djl.ai/dataset/nlp/ai/djl/basicdataset/amazon_reviews/1.0/amazon_reviews_us_Digital_Software_v1_00.tsv.gz";
        float paddingToken = tokenizer.getVocabulary().getIndex("[PAD]");
        return CsvDataset.builder()
                .optCsvUrl(amazonReview)
                .setCsvFormat(CSVFormat.TDF.builder().setQuote(null).setHeader().build())
                .setSampling(arguments.getBatchSize(), true)
                .addFeature(new Feature("review_body", new BertFeaturizer(tokenizer, maxLength)))
                .addLabel(
                        new Feature(
                                "star_rating",
                                (buf, data) -> buf.put(Float.parseFloat(data) - 1.0f)))
                .optDataBatchifier(
                        PaddingStackBatchifier.builder()
                                .optIncludeValidLengths(false)
                                .addPad(0, 0, (m) -> m.ones(new Shape(1)).mul(paddingToken))
                                .build())
                .optLimit(arguments.getLimit())
                .build();
    }

    private static Block getBlock(Block embedder) {
        SequentialBlock classifier = new SequentialBlock();
        // text embedding layer
        LambdaBlock lambda =
                new LambdaBlock(
                        ndList -> {
                            NDArray data = ndList.singletonOrThrow();
                            NDList inputs = new NDList();
                            inputs.add(data.toType(DataType.INT64, false));
                            inputs.add(data.getManager().full(data.getShape(), 1, DataType.INT64));
                            inputs.add(
                                    data.getManager()
                                            .arange(data.getShape().get(1)) // maxLen
                                            .toType(DataType.INT64, false)
                                            .broadcast(data.getShape()));
                            return inputs;
                        });
        classifier.add(lambda);
        classifier.add(embedder);
        // Classification layers
        classifier
                .add(Linear.builder().setUnits(768).build()) // pre classifier
                .add(Activation::relu)
                .add(Dropout.builder().optRate(0.2f).build())
                .add(Linear.builder().setUnits(5).build()) // 5 star rating
                .addSingleton(nd -> nd.get(":,0")); // follow HF classifier
        return classifier;
    }

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
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
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optDevices(arguments.getMaxGpus())
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    private static final class BertFeaturizer implements DataFeaturizer {

        private final BertFullTokenizer tokenizer;
        private final int maxLength;

        public BertFeaturizer(BertFullTokenizer tokenizer, int maxLength) {
            this.tokenizer = tokenizer;
            this.maxLength = maxLength;
        }

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            Vocabulary vocab = tokenizer.getVocabulary();
            List<String> tokens = tokenizer.tokenize(input.toLowerCase(Locale.ENGLISH));
            tokens = tokens.size() > maxLength ? tokens.subList(0, maxLength) : tokens;
            buf.put(vocab.getIndex("[CLS]"));
            tokens.forEach(token -> buf.put(vocab.getIndex(token)));
            buf.put(vocab.getIndex("[SEP]"));
        }
    }
}
