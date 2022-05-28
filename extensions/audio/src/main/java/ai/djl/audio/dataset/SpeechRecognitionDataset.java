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

package ai.djl.audio.dataset;

import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.nlp.TextDataset;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.engine.Engine;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.RandomAccessDataset;
import java.util.List;

/**
 * {@code SpeechRecognitionDataset} is an abstract dataset that can be used for datasets for
 * Automatic Speech Recognition (ASR) where the source data is {@link AudioData} and the target data
 * is {@link TextData}.
 *
 * <p>For the target data, it will create embeddings for the target data. Embeddings can be either
 * pre-trained or trained on the go. Pre-trained {@link TextEmbedding} must be set in the {@link
 * TextDataset.Builder}. If no embeddings are set, the dataset creates {@link
 * TrainableWordEmbedding} based {@link TrainableWordEmbedding} from the {@link Vocabulary} created
 * within the dataset.
 *
 * <p>For the source data, it will use the {@link ai.djl.audio.processor.AudioProcessor} to
 * featurize data, if users want to write their own featurizer, they can get the original {@link
 * NDArray} from {@link AudioData} without using any {@link ai.djl.audio.processor.AudioProcessor}.
 */
public abstract class SpeechRecognitionDataset extends RandomAccessDataset {

    protected AudioData sourceAudioData;
    protected TextData targetTextData;
    protected NDManager manager;
    protected Usage usage;

    protected MRL mrl;
    protected boolean prepared;

    /**
     * Creates a new instance of {@link SpeechRecognitionDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public SpeechRecognitionDataset(AudioBuilder<?> builder) {
        super(builder);
        sourceAudioData =
                new AudioData(getDefaultConfiguration().update(builder.sourceConfiguration));
        targetTextData =
                new TextData(
                        TextData.getDefaultConfiguration().update(builder.targetConfiguration));
        manager = builder.manager;
        usage = builder.usage;
    }

    /**
     * @param newTextData list of all unprocessed sentences in the dataset.
     * @throws EmbeddingException if there is an error while embedding input.
     */
    protected void targetPreprocess(List<String> newTextData) throws EmbeddingException {
        TextData textData = targetTextData;
        textData.preprocess(
                manager, newTextData.subList(0, (int) Math.min(limit, newTextData.size())));
    }

    /**
     * This method is used to set the audio data path.
     *
     * @param audioPathList The path list of all original audio data
     */
    protected void sourcePreprocess(List<String> audioPathList) {
        sourceAudioData.setAudioPaths(audioPathList);
    }

    /** @return A default {@link ai.djl.audio.dataset.AudioData.Configuration}. */
    public static AudioData.Configuration getDefaultConfiguration() {
        return AudioData.getDefaultConfiguration();
    }

    /** Abstract AudioBuilder that helps build a {@link SpeechRecognitionDataset}. */
    public abstract static class AudioBuilder<T extends AudioBuilder<T>> extends BaseBuilder<T> {

        protected AudioData.Configuration sourceConfiguration;
        protected TextData.Configuration targetConfiguration;
        protected NDManager manager;

        protected Repository repository;
        protected String groupId;
        protected Usage usage;
        protected String artifactId;

        /** Constructs a new builder. */
        AudioBuilder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            usage = Usage.TRAIN;
            sourceConfiguration = new AudioData.Configuration();
            targetConfiguration = new TextData.Configuration();
            manager = Engine.getInstance().newBaseManager();
        }

        /**
         * Sets the {@link AudioData.Configuration} to use for the source text data.
         *
         * @param sourceConfiguration the {@link AudioData.Configuration}
         * @return this builder
         */
        public T setSourceConfiguration(AudioData.Configuration sourceConfiguration) {
            this.sourceConfiguration = sourceConfiguration;
            return self();
        }

        /**
         * Sets the {@link TextData.Configuration} to use for the target text data.
         *
         * @param targetConfiguration the {@link TextData.Configuration}
         * @return this builder
         */
        public T setTargetConfiguration(TextData.Configuration targetConfiguration) {
            this.targetConfiguration = targetConfiguration;
            return self();
        }

        /**
         * Sets the optional manager for the dataset (default follows engine default).
         *
         * @param manager the manager
         * @return this builder
         */
        public T optManager(NDManager manager) {
            this.manager = manager.newSubManager();
            return self();
        }

        /**
         * Sets the optional usage.
         *
         * @param usage the usage
         * @return this builder
         */
        public T optUsage(Usage usage) {
            this.usage = usage;
            return self();
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
         * @return this builder
         */
        public T optRepository(Repository repository) {
            this.repository = repository;
            return self();
        }

        /**
         * Sets optional groupId.
         *
         * @param groupId the groupId}
         * @return this builder
         */
        public T optGroupId(String groupId) {
            this.groupId = groupId;
            return self();
        }
    }
}
