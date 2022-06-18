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
package ai.djl.basicdataset.nlp;

import ai.djl.Application;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.RawDataset;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;

import java.io.IOException;
import java.nio.file.Path;

/**
 * The WikiText language modeling dataset is a collection of over 100 million tokens extracted from
 * the set of verified Good and Featured articles on Wikipedia.
 */
public class WikiText2 implements RawDataset<Path> {

    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "wikitext-2";

    private Dataset.Usage usage;
    private Path root;

    private MRL mrl;
    private boolean prepared;

    WikiText2(Builder builder) {
        this.usage = builder.usage;
        mrl = builder.getMrl();
    }

    /**
     * Creates a builder to build a {@link WikiText2}.
     *
     * @return a new {@link WikiText2.Builder} object
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Prepares the dataset for use with tracked progress.
     *
     * @param progress the progress tracker
     * @throws IOException for various exceptions depending on the dataset
     */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);
        Artifact.Item item;
        item = artifact.getFiles().get("wikitext-2");
        String path;
        switch (usage) {
            case TRAIN:
                path = "wikitext-2/wiki.train.tokens";
                break;
            case TEST:
                path = "wikitext-2/wiki.test.tokens";
                break;
            case VALIDATION:
                path = "wikitext-2/wiki.valid.tokens";
                break;
            default:
                throw new UnsupportedOperationException("Unsupported usage type.");
        }

        root = mrl.getRepository().getFile(item, path).toAbsolutePath();
        prepared = true;
    }

    /**
     * Fetches an iterator that can iterate through the {@link Dataset}. This method is not
     * implemented for the WikiText2 dataset because the WikiText2 dataset is not suitable for
     * iteration. If the method is called, it will directly return {@code null}.
     *
     * @param manager the dataset to iterate through
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     */
    @Override
    public Iterable<Batch> getData(NDManager manager) throws IOException, TranslateException {
        return null;
    }

    /**
     * Get data from the WikiText2 dataset. This method will directly return the whole dataset.
     *
     * @return a {@link Path} object locating the WikiText2 dataset file
     */
    @Override
    public Path getData() throws IOException {
        prepare(null);
        return root;
    }

    /** A builder to construct a {@link WikiText2} . */
    public static final class Builder {

        Repository repository;
        String groupId;
        String artifactId;
        Dataset.Usage usage;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Dataset.Usage.TRAIN;
        }

        /**
         * Sets the optional repository for the dataset.
         *
         * @param repository the new repository
         * @return this builder
         */
        public Builder optRepository(Repository repository) {
            this.repository = repository;
            return this;
        }

        /**
         * Sets optional groupId.
         *
         * @param groupId the groupId
         * @return this builder
         */
        public Builder optGroupId(String groupId) {
            this.groupId = groupId;
            return this;
        }

        /**
         * Sets the optional artifactId.
         *
         * @param artifactId the artifactId
         * @return this builder
         */
        public Builder optArtifactId(String artifactId) {
            if (artifactId.contains(":")) {
                String[] tokens = artifactId.split(":");
                groupId = tokens[0];
                this.artifactId = tokens[1];
            } else {
                this.artifactId = artifactId;
            }
            return this;
        }

        /**
         * Sets the optional usage for the dataset.
         *
         * @param usage the usage
         * @return this builder
         */
        public Builder optUsage(Dataset.Usage usage) {
            this.usage = usage;
            return this;
        }

        /**
         * Builds a new {@link WikiText2} object.
         *
         * @return the new {@link WikiText2} object
         */
        public WikiText2 build() {
            return new WikiText2(this);
        }

        MRL getMrl() {
            return repository.dataset(Application.NLP.ANY, groupId, artifactId, VERSION);
        }
    }
}
