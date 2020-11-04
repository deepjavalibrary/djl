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

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.CsvDataset;
import ai.djl.basicdataset.utils.DynamicBuffer;
import ai.djl.examples.training.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DataManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.CheckpointsTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.apache.commons.csv.CSVFormat;

public final class TrainAmazonReviewRanking {

    private TrainAmazonReviewRanking() {}

    public static void main(String[] args)
            throws IOException, TranslateException, ModelNotFoundException,
                    MalformedModelException {
        TrainAmazonReviewRanking.runExample(args);
    }

    public static TrainingResult runExample(String[] args)
            throws IOException, TranslateException, ModelNotFoundException,
                    MalformedModelException {
        Arguments arguments = Arguments.parseArgs(args);
        if (arguments == null) {
            return null;
        }

        NoopTranslator translator = new NoopTranslator();
        translator.setBatchifier(null);
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.WORD_EMBEDDING)
                        .setTypes(NDList.class, NDList.class)
                        .optTranslator(translator)
                        .optModelUrls(
                                "https://alpha-djl-demos.s3.amazonaws.com/model/examples/distilbert.zip")
                        .optProgress(new ProgressBar())
                        .build();
        try (Model model = Model.newInstance("AmazonReviewRatingClassification");
                ZooModel<NDList, NDList> embedding = ModelZoo.loadModel(criteria)) {
            // Prepare dataset
            BertFullTokenizer tokenizer =
                    new BertFullTokenizer(embedding.getArtifact("vocab.txt").getPath(), true);
            CsvDataset amazonReviewDataset = getDataset(arguments, tokenizer);
            // split data with 7:3 train:valid ratio
            RandomAccessDataset[] datasets = amazonReviewDataset.randomSplit(7, 3);
            RandomAccessDataset trainingSet = datasets[0];
            RandomAccessDataset validationSet = datasets[1];
            // create classification layer
            model.setBlock(getClassifier());
            DefaultTrainingConfig config = setupTrainingConfig(arguments, embedding.newPredictor());
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());
                Shape encoderInputShape = new Shape(arguments.getBatchSize(), 10, 768);

                // initialize trainer with proper input shape
                trainer.initialize(encoderInputShape);

                EasyTrain.fit(trainer, arguments.getEpoch(), trainingSet, validationSet);
                return trainer.getTrainingResult();
            }
        }
    }

    private static CsvDataset getDataset(Arguments arguments, BertFullTokenizer tokenizer) {
        String amazonReview =
                "https://github.com/data-science-on-aws/workshop/raw/master/07_train/data/amazon_reviews_us_Digital_Software_v1_00.tsv.gz";
        float paddingToken = tokenizer.getVocabulary().getIndex("[PAD]");
        return CsvDataset.builder()
                .optCsvUrl(amazonReview)
                .setCsvFormat(CSVFormat.TDF.withQuote(null).withHeader())
                .setSampling(arguments.getBatchSize(), true)
                .addFeature(new CsvDataset.Feature("review_body", new BertFeaturizer(tokenizer)))
                .addNumericLabel("star_rating")
                .optDataBatchifier(
                        PaddingStackBatchifier.builder()
                                .optIncludeValidLengths(false)
                                .addPad(0, 0, (m) -> m.ones(new Shape(1)).mul(paddingToken))
                                .build())
                .build();
    }

    private static Block getClassifier() {
        return new SequentialBlock()
                .add(Linear.builder().setUnits(768).build()) // pre classifier
                .add(Activation::relu)
                .add(Dropout.builder().optRate(0.2f).build())
                .add(Linear.builder().setUnits(5).build()) // 5 star rating
                .add(
                        list ->
                                new NDList(
                                        list.singletonOrThrow()
                                                .get(":,0"))); // follow HF classifier
    }

    private static DefaultTrainingConfig setupTrainingConfig(
            Arguments arguments, Predictor<NDList, NDList> predictor) {
        String outputDir = arguments.getOutputDir();
        CheckpointsTrainingListener listener = new CheckpointsTrainingListener(outputDir);
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
                .optDevices(Device.getDevices(1))
                .optDataManager(new EmbeddingDataManager(predictor))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    private static final class BertFeaturizer implements CsvDataset.Featurizer {

        private final BertFullTokenizer tokenizer;

        public BertFeaturizer(BertFullTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            SimpleVocabulary vocab = tokenizer.getVocabulary();
            tokenizer.tokenize(input).forEach(token -> buf.put(vocab.getIndex(token)));
        }
    }

    private static final class EmbeddingDataManager extends DataManager {

        private final Predictor<NDList, NDList> embedding;

        public EmbeddingDataManager(Predictor<NDList, NDList> embedding) {
            this.embedding = embedding;
        }

        /** {@inheritDoc} */
        @Override
        public NDList getData(Batch batch) {
            try {
                // batch shape (batchsize, maximum length)
                NDArray data = batch.getData().head();
                long batchsize = data.getShape().get(0);
                float maxLength = data.getShape().get(1);
                return embedding.predict(
                        new NDList(data, data.getManager().full(new Shape(batchsize), maxLength)));
            } catch (TranslateException e) {
                throw new IllegalArgumentException(e.getMessage(), e);
            }
        }
    }
}
