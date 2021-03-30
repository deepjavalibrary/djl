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
package ai.djl.fasttext;

import ai.djl.Device;
import ai.djl.nn.Parameter;
import ai.djl.training.TrainingConfig;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.PairList;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.function.Predicate;

/** An interface that is responsible for holding the configuration required by fastText training. */
public class FtTrainingConfig implements TrainingConfig {

    private FtTrainingMode trainingMode;
    private Path outputDir;
    private String modelName;
    private int epoch;
    private int minWordCount;
    private int minLabelCount;
    private int maxNgramLength;
    private int minCharLength;
    private int maxCharLength;
    private int bucket;
    private float samplingThreshold;
    private String labelPrefix;
    private float learningRate;
    private int learningRateUpdateRate;
    private int wordVecSize;
    private int contextWindow;
    private int numNegativesSampled;
    private int threads;
    private String loss;

    FtTrainingConfig(Builder builder) {
        trainingMode = builder.trainingMode;
        outputDir = builder.outputDir;
        modelName = builder.modelName;
        epoch = builder.epoch;
        minWordCount = builder.minWordCount;
        minLabelCount = builder.minLabelCount;
        maxNgramLength = builder.maxNgramLength;
        minCharLength = builder.minCharLength;
        maxCharLength = builder.maxCharLength;
        bucket = builder.bucket;
        samplingThreshold = builder.samplingThreshold;
        labelPrefix = builder.labelPrefix;
        learningRate = builder.learningRate;
        learningRateUpdateRate = builder.learningRateUpdateRate;
        wordVecSize = builder.wordVecSize;
        contextWindow = builder.contextWindow;
        numNegativesSampled = builder.numNegativesSampled;
        threads = builder.threads;
        loss = builder.loss;
    }

    /**
     * Returns the @{link FtTrainingMode}.
     *
     * @return the @{code FtTrainingMode}
     */
    public FtTrainingMode getTrainingMode() {
        return trainingMode;
    }

    /**
     * Returns the output directory.
     *
     * @return the output directory
     */
    public Path getOutputDir() {
        return outputDir;
    }

    /**
     * Return the name of the model.
     *
     * @return the name of the model
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * Returns number of epochs.
     *
     * @return number of epochs
     */
    public int getEpoch() {
        return epoch;
    }

    /**
     * Return minimal number of word occurrences.
     *
     * @return minimal number of word occurrences
     */
    public int getMinWordCount() {
        return minWordCount;
    }

    /**
     * Returns minimal number of label occurrences.
     *
     * @return minimal number of label occurrences
     */
    public int getMinLabelCount() {
        return minLabelCount;
    }

    /**
     * Returns maximum length of word ngram.
     *
     * @return maximum length of word ngram
     */
    public int getMaxNgramLength() {
        return maxNgramLength;
    }

    /**
     * Return minimum length of char ngram.
     *
     * @return minimum length of char ngram
     */
    public int getMinCharLength() {
        return minCharLength;
    }

    /**
     * Return maximum length of char ngram.
     *
     * @return maximum length of char ngram
     */
    public int getMaxCharLength() {
        return maxCharLength;
    }

    /**
     * Returns number of buckets.
     *
     * @return number of buckets
     */
    public int getBucket() {
        return bucket;
    }

    /**
     * Returns sampling threshold.
     *
     * @return sampling threshold
     */
    public float getSamplingThreshold() {
        return samplingThreshold;
    }

    /**
     * Return label prefix.
     *
     * @return label prefix
     */
    public String getLabelPrefix() {
        return labelPrefix;
    }

    /**
     * Returns learning rate.
     *
     * @return learning rate
     */
    public float getLearningRate() {
        return learningRate;
    }

    /**
     * Returns the rate of updates for the learning rate.
     *
     * @return the rate of updates for the learning rate
     */
    public int getLearningRateUpdateRate() {
        return learningRateUpdateRate;
    }

    /**
     * Returns size of word vectors.
     *
     * @return size of word vectors
     */
    public int getWordVecSize() {
        return wordVecSize;
    }

    /**
     * Returns size of the context window.
     *
     * @return size of the context window
     */
    public int getContextWindow() {
        return contextWindow;
    }

    /**
     * Returns number of negatives sampled.
     *
     * @return number of negatives sampled
     */
    public int getNumNegativesSampled() {
        return numNegativesSampled;
    }

    /**
     * Returns number training threads.
     *
     * @return number training threads
     */
    public int getThreads() {
        return threads;
    }

    /**
     * Returns the loss function.
     *
     * @return the loss function
     */
    public String getLoss() {
        return loss;
    }

    /** {@inheritDoc} */
    @Override
    public Device[] getDevices() {
        return new Device[0];
    }

    /** {@inheritDoc} */
    @Override
    public PairList<Initializer, Predicate<Parameter>> getInitializers() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Optimizer getOptimizer() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Loss getLossFunction() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public ExecutorService getExecutorService() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public List<Evaluator> getEvaluators() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public List<TrainingListener> getTrainingListeners() {
        return null;
    }

    /**
     * Returns the fastText command in an array.
     *
     * @param input training dataset file path
     * @return the fastText command in an array
     */
    public String[] toCommand(String input) {
        Path modelFile = outputDir.resolve(modelName).toAbsolutePath();

        List<String> cmd = new ArrayList<>();
        cmd.add("fasttext");
        cmd.add(trainingMode.name().toLowerCase());
        cmd.add("-input");
        cmd.add(input);
        cmd.add("-output");
        cmd.add(modelFile.toString());
        if (epoch >= 0) {
            cmd.add("-epoch");
            cmd.add(String.valueOf(epoch));
        }
        if (minWordCount >= 0) {
            cmd.add("-minCount");
            cmd.add(String.valueOf(minWordCount));
        }
        if (minLabelCount >= 0) {
            cmd.add("-minCountLabel");
            cmd.add(String.valueOf(minLabelCount));
        }
        if (maxNgramLength >= 0) {
            cmd.add("-wordNgrams");
            cmd.add(String.valueOf(maxNgramLength));
        }
        if (minCharLength >= 0) {
            cmd.add("-minn");
            cmd.add(String.valueOf(minCharLength));
        }
        if (maxCharLength >= 0) {
            cmd.add("-maxn");
            cmd.add(String.valueOf(maxCharLength));
        }
        if (bucket >= 0) {
            cmd.add("-bucket");
            cmd.add(String.valueOf(bucket));
        }
        if (samplingThreshold >= 0) {
            cmd.add("-t");
            cmd.add(String.valueOf(samplingThreshold));
        }
        if (labelPrefix != null) {
            cmd.add("-label");
            cmd.add(labelPrefix);
        }
        if (learningRate >= 0) {
            cmd.add("-lr");
            cmd.add(String.valueOf(learningRate));
        }
        if (learningRateUpdateRate >= 0) {
            cmd.add("-lrUpdateRate");
            cmd.add(String.valueOf(learningRateUpdateRate));
        }
        if (wordVecSize >= 0) {
            cmd.add("-dim");
            cmd.add(String.valueOf(wordVecSize));
        }
        if (contextWindow >= 0) {
            cmd.add("-ws");
            cmd.add(String.valueOf(contextWindow));
        }
        if (numNegativesSampled >= 0) {
            cmd.add("-neg");
            cmd.add(String.valueOf(numNegativesSampled));
        }
        if (threads >= 0) {
            cmd.add("-thread");
            cmd.add(String.valueOf(threads));
        }
        if (loss != null) {
            cmd.add("-loss");
            cmd.add(loss);
        }
        return cmd.toArray(new String[0]);
    }

    /**
     * Creates a builder to build a {@code FtTrainingConfig}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A builder to construct a {@code FtTrainingConfig}. */
    public static final class Builder {

        FtTrainingMode trainingMode = FtTrainingMode.SUPERVISED;
        Path outputDir;
        String modelName;
        int epoch = -1;
        int minWordCount = -1;
        int minLabelCount = -1;
        int maxNgramLength = -1;
        int minCharLength = -1;
        int maxCharLength = -1;
        int bucket = -1;
        float samplingThreshold = -1f;
        String labelPrefix;
        float learningRate = -1f;
        int learningRateUpdateRate = -1;
        int wordVecSize = -1;
        int contextWindow = -1;
        int numNegativesSampled = -1;
        int threads = -1;
        String loss;

        Builder() {}

        /**
         * Sets the output directory.
         *
         * @param outputDir the output directory
         * @return this builder
         */
        public Builder setOutputDir(Path outputDir) {
            this.outputDir = outputDir;
            return this;
        }

        /**
         * Sets the name of the model.
         *
         * @param modelName the name of the model
         * @return this builder
         */
        public Builder setModelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * Sets the optional {@link FtTrainingMode}.
         *
         * @param trainingMode the optional {@code FtTrainingMode} (default {@link
         *     FtTrainingMode#SUPERVISED}
         * @return this builder
         */
        public Builder optTrainingMode(FtTrainingMode trainingMode) {
            this.trainingMode = trainingMode;
            return this;
        }

        /**
         * Sets the optional number of epochs.
         *
         * @param epoch the optional number of epochs (default 5)
         * @return this builder
         */
        public Builder optEpoch(int epoch) {
            this.epoch = epoch;
            return this;
        }

        /**
         * Sets the optional minimal number of word occurrences.
         *
         * @param minWordCount the optional minimal number of word occurrences (default 1)
         * @return this builder
         */
        public Builder optMinWordCount(int minWordCount) {
            this.minWordCount = minWordCount;
            return this;
        }

        /**
         * Sets the optional minimal number of label occurrences.
         *
         * @param minLabelCount the optional minimal number of label occurrences (default 0)
         * @return this builder
         */
        public Builder optMinLabelCount(int minLabelCount) {
            this.minLabelCount = minLabelCount;
            return this;
        }

        /**
         * Sets the optional maximum length of word ngram.
         *
         * @param maxNgramLength the optional maximum length of word ngram (default 1)
         * @return this builder
         */
        public Builder optMaxNGramLength(int maxNgramLength) {
            this.maxNgramLength = maxNgramLength;
            return this;
        }

        /**
         * Sets the optional minimum length of char ngram.
         *
         * @param minCharLength the optional minimum length of char ngram (default 0)
         * @return this builder
         */
        public Builder optMinCharLength(int minCharLength) {
            this.minCharLength = minCharLength;
            return this;
        }

        /**
         * Sets the optional maximum length of char ngram.
         *
         * @param maxCharLength the optional maximum length of char ngram (default 0)
         * @return this builder
         */
        public Builder optMaxCharLength(int maxCharLength) {
            this.maxCharLength = maxCharLength;
            return this;
        }

        /**
         * Sets the optional number of buckets.
         *
         * @param bucket the optional number of buckets (default 2000000)
         * @return this builder
         */
        public Builder optBucket(int bucket) {
            this.bucket = bucket;
            return this;
        }

        /**
         * Sets the optional sampling threshold.
         *
         * @param samplingThreshold the optional sampling threshold (default 0.0001)
         * @return this builder
         */
        public Builder optSamplingThreshold(float samplingThreshold) {
            this.samplingThreshold = samplingThreshold;
            return this;
        }

        /**
         * Sets the optional label prefix.
         *
         * @param labelPrefix the optional label prefix (default "__lable__")
         * @return this builder
         */
        public Builder optLabelPrefix(String labelPrefix) {
            this.labelPrefix = labelPrefix;
            return this;
        }

        /**
         * Sets the optional learning rate.
         *
         * @param learningRate the optional learning rate (default 0.1)
         * @return this builder
         */
        public Builder optLearningRate(float learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        /**
         * Sets the optional rate of updates for the learning rate.
         *
         * @param learningRateUpdateRate the optional rate of updates for the learning rate (default
         *     100)
         * @return this builder
         */
        public Builder optLearningRateUpdateRate(int learningRateUpdateRate) {
            this.learningRateUpdateRate = learningRateUpdateRate;
            return this;
        }

        /**
         * Sets the optional size of word vectors.
         *
         * @param wordVecSize the optional size of word vectors (default 100)
         * @return this builder
         */
        public Builder optWordVecSize(int wordVecSize) {
            this.wordVecSize = wordVecSize;
            return this;
        }

        /**
         * Sets the optional size of the context window.
         *
         * @param contextWindow the optional size of the context window (default 5)
         * @return this builder
         */
        public Builder optContextWindow(int contextWindow) {
            this.contextWindow = contextWindow;
            return this;
        }

        /**
         * Sets the optional number of negatives sampled.
         *
         * @param numNegativesSampled the optional number of negatives sampled (default 5)
         * @return this builder
         */
        public Builder optNumNegativesSampled(int numNegativesSampled) {
            this.numNegativesSampled = numNegativesSampled;
            return this;
        }

        /**
         * Sets the optional number training threads.
         *
         * @param threads the optional number training threads (default 12)
         * @return this builder
         */
        public Builder optThreads(int threads) {
            this.threads = threads;
            return this;
        }

        /**
         * Sets the optional loss function.
         *
         * @param loss the optional loss function (default {@link FtLoss#SOFTMAX}
         * @return this builder
         */
        public Builder optLoss(FtLoss loss) {
            this.loss = loss.name().toLowerCase();
            return this;
        }

        /**
         * Builds a new {@code FtTrainingConfig}.
         *
         * @return the new {@code FtTrainingConfig}
         */
        public FtTrainingConfig build() {
            return new FtTrainingConfig(this);
        }
    }

    /** Loss functions that fastText supports. */
    public enum FtLoss {
        NS,
        HS,
        SOFTMAX,
        OVA
    }
}
