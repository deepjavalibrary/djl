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
package ai.djl.fasttext.dataset;

import ai.djl.repository.MRL;
import ai.djl.repository.Repository;

/**
 * A text classification dataset contains questions from cooking.stackexchange.com and their
 * associated tags on the site.
 */
public class CookingStackExchange extends FtDataset {

    private static final String ARTIFACT_ID = "cooking_stackexchange";

    CookingStackExchange(Builder builder) {
        this.repository = builder.repository;
        this.usage = builder.usage;
    }

    /**
     * Creates a builder to build a {@code CookingStackExchange}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    public MRL getMrl() {
        return new MRL(MRL.Dataset.NLP, FtDatasets.GROUP_ID, ARTIFACT_ID);
    }

    /** A builder to construct a {@link CookingStackExchange}. */
    public static final class Builder {

        Repository repository;
        Usage usage;

        /** Constructs a new builder. */
        Builder() {
            repository = FtDatasets.REPOSITORY;
            usage = Usage.TRAIN;
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
         * Sets the optional usage for the dataset.
         *
         * @param usage the usage
         * @return this builder
         */
        public Builder optUsage(Usage usage) {
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
    }
}
