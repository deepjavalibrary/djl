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
import ai.djl.basicdataset.utils.TextData;
import ai.djl.engine.Engine;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.RandomAccessDataset;
import java.util.List;

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

    protected void targetPreprocess(List<String> newTextData, boolean source)
            throws EmbeddingException {
        TextData textData = targetTextData;
        textData.preprocess(
                manager, newTextData.subList(0, (int) Math.min(limit, newTextData.size())));
    }

    public NDArray getProcessedAudioData(String path) {
        AudioData audioData = sourceAudioData;
        return audioData.getPreprocessedData(manager, path);
    }

    public static AudioData.Configuration getDefaultConfiguration() {
        return new AudioData.Configuration();
    }

    /** Abstract AudioBuilder that helps build a {@link SpeechRecognitionDataset}. */
    public abstract static class AudioBuilder<T extends AudioBuilder<T>> extends BaseBuilder<T> {

        protected AudioData.Configuration sourceConfiguration;
        protected TextData.Configuration targetConfiguration;
        protected NDManager manager;

        protected Repository repository;
        protected String groupId;
        // protected String artifactId;
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
