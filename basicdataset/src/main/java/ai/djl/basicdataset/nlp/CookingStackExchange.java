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
package ai.djl.basicdataset.nlp;

import ai.djl.Application.NLP;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.RawDataset;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.util.Progress;
import java.io.IOException;
import java.nio.file.Path;

/**
 * A text classification dataset contains questions from cooking.stackexchange.com and their
 * associated tags on the site.
 */
public class CookingStackExchange implements RawDataset<Path> {

    private static final String ARTIFACT_ID = "cooking_stackexchange";
    private static final String VERSION = "1.0";

    private Dataset.Usage usage;
    private Path root;

    private MRL mrl;
    private boolean prepared;

    CookingStackExchange(Builder builder) {
        this.usage = builder.usage;
        mrl = builder.getMrl();
    }

    /** {@inheritDoc} */
    @Override
    public Path getData() throws IOException {
        prepare(null);
        return root;
    }

    /** {@inheritDoc} */
    @Override
    public Iterable<Batch> getData(NDManager manager) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);

        Artifact.Item item;
        switch (usage) {
            case TRAIN:
                item = artifact.getFiles().get("train");
                break;
            case TEST:
                item = artifact.getFiles().get("test");
                break;
            case VALIDATION:
            default:
                throw new IOException("Only training and testing dataset supported.");
        }
        root = mrl.getRepository().getFile(item, "").toAbsolutePath();
        prepared = true;
    }

    /**
     * Creates a builder to build a {@code CookingStackExchange}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A builder to construct a {@link CookingStackExchange}. */
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
         * @param groupId the groupId}
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
         * Builds a new {@code CookingStackExchange}.
         *
         * @return the new {@code CookingStackExchange}
         */
        public CookingStackExchange build() {
            return new CookingStackExchange(this);
        }

        MRL getMrl() {
            return repository.dataset(NLP.ANY, groupId, artifactId, VERSION);
        }
    }
}
