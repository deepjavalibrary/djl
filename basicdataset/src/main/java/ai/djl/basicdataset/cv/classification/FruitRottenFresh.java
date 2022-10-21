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
package ai.djl.basicdataset.cv.classification;

import ai.djl.Application;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.translate.Pipeline;
import ai.djl.util.Progress;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * FruitRottenFresh classification dataset that contains the same fruit where rotten and fresh class
 * are stored in two subfolders. It is structured similar to ImageFolders as follows:
 * root/shoes/Aerobic Shoes1.png root/shoes/Aerobic Shose2.png ... root/boots/Black Boots.png
 * root/boots/White Boots.png ... root/pumps/Red Pumps.png root/pumps/Pink Pumps.png ...
 */
public final class FruitRottenFresh extends AbstractImageFolder {

    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "fruit-unittest";

    private MRL mrl;
    private boolean prepared;

    private FruitRottenFresh(Builder builder) {
        super(builder);
        mrl = builder.getMrl();
    }

    /**
     * Creates a new builder to build a {@link FruitRottenFresh}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    protected Path getImagePath(String key) {
        return Paths.get(key);
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {

        if (prepared) {
            return;
        }

        // Use the code in ImageFolder
        mrl.prepare(null, progress);
        loadSynset();
        Path root = Paths.get(mrl.getRepository().getBaseUri());
        if (progress != null) {
            progress.reset("Preparing", 2);
            progress.start(0);
            listImages(root, synset);
            progress.end();
        } else {
            listImages(root, synset);
        }

        prepared = true;
    }

    private void loadSynset() {
        File root = new File(mrl.getRepository().getBaseUri());
        File[] dir = root.listFiles(f -> f.isDirectory() && !f.getName().startsWith("."));
        if (dir == null || dir.length == 0) {
            throw new IllegalArgumentException(root + " not found or didn't have any file in it");
        }
        Arrays.sort(dir);
        for (File file : dir) {
            synset.add(file.getName());
        }
    }

    /** A builder for the {@link FruitRottenFresh}. */
    public static final class Builder extends ImageFolderBuilder<Builder> {

        String groupId;
        String artifactId;
        Usage usage;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Usage.TRAIN;
        }

        /** {@inheritDoc} */
        @Override
        public Builder self() {
            return this;
        }

        /**
         * Sets the optional usage.
         *
         * @param usage the usage
         * @return this builder
         */
        public Builder optUsage(Usage usage) {
            this.usage = usage;
            return self();
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
         * @return this builder
         */
        public Builder optRepository(Repository repository) {
            this.repository = repository;
            return self();
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
         * Builds the {@link FruitRottenFresh}.
         *
         * @return the {@link FruitRottenFresh}
         * @throws IOException if there is an issue
         */
        public FruitRottenFresh build() throws IOException {
            if (pipeline == null) {
                pipeline = new Pipeline(new ToTensor());
            }

            MRL mrl = getMrl();
            Artifact artifact = mrl.getDefaultArtifact();
            // Downloading the cache happens here
            mrl.prepare(artifact, null);

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
            Path root = mrl.getRepository().getFile(item, "").toAbsolutePath();

            repository = Repository.newInstance("banana", root);
            return new FruitRottenFresh(this);
        }

        MRL getMrl() {
            return repository.dataset(Application.CV.ANY, groupId, artifactId, VERSION);
        }
    }
}
