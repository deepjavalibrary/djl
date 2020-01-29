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
package ai.djl.basicdataset;

import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.modality.cv.util.NDImageUtils.Flag;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.dataset.ZooDataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.Pipeline;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link ai.djl.training.dataset.Dataset} featuring captcha images.
 *
 * <p>Each image is a 160x60 grayscale image featuring 5 or 6 digits where each digit ranges from
 * 0-10. The dataset therefore features 6 labels. Each label ranges from 0-11 where 0-10 represent a
 * recognized digit and 11 indicates that the value is not a digit (size 5 and not 6).
 */
public class CaptchaDataset extends RandomAccessDataset implements ZooDataset {

    public static final int IMAGE_WIDTH = 160;
    public static final int IMAGE_HEIGHT = 60;
    public static final int CAPTCHA_LENGTH = 6;
    public static final int CAPTCHA_OPTIONS = 11;

    private static final String ARTIFACT_ID = "captcha";

    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;

    private List<String> items;

    /**
     * Creates a new instance of {@link CaptchaDataset}.
     *
     * @param builder a builder with the necessary configurations
     */
    public CaptchaDataset(Builder builder) {
        super(builder);
        this.repository = builder.repository;
        this.artifact = builder.artifact;
        this.usage = builder.usage;
    }

    /**
     * Creates a builder to build a {@link CaptchaDataset}.
     *
     * @return a new builder
     */
    public static CaptchaDataset.Builder builder() {
        return new CaptchaDataset.Builder();
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) throws IOException {
        String item = items.get(Math.toIntExact(index));

        Path imagePath =
                repository.getFile(getArtifactItem(), getUsagePath() + "/" + item + ".jpeg");
        NDArray imageArray = BufferedImageUtils.readFileToArray(manager, imagePath, Flag.GRAYSCALE);
        NDList data = new NDList(imageArray);

        NDList labels = new NDList(CAPTCHA_LENGTH);
        char[] labelChars = item.toCharArray();
        for (int i = 0; i < CAPTCHA_LENGTH; i++) {
            if (i < item.length()) {
                int labelDigit = Integer.parseInt(Character.toString(labelChars[i]));
                labels.add(manager.create(labelDigit));
            } else {
                labels.add(manager.create(11));
            }
        }

        return new Record(data, labels);
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return items.size();
    }

    /** {@inheritDoc} */
    @Override
    public MRL getMrl() {
        return new MRL(MRL.Dataset.CV, BasicDatasets.GROUP_ID, ARTIFACT_ID);
    }

    /** {@inheritDoc} */
    @Override
    public Repository getRepository() {
        return repository;
    }

    /** {@inheritDoc} */
    @Override
    public Artifact getArtifact() {
        return artifact;
    }

    /** {@inheritDoc} */
    @Override
    public Usage getUsage() {
        return usage;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isPrepared() {
        return prepared;
    }

    /** {@inheritDoc} */
    @Override
    public void setPrepared(boolean prepared) {
        this.prepared = prepared;
    }

    /** {@inheritDoc} */
    @Override
    public void useDefaultArtifact() throws IOException {
        artifact = repository.resolve(getMrl(), "1.0", null);
    }

    /** {@inheritDoc} */
    @Override
    public void prepareData(Usage usage) throws IOException {

        items = new ArrayList<>();
        for (String filenameWithExtension :
                repository.listDirectory(getArtifactItem(), getUsagePath())) {
            String captchaFilename =
                    filenameWithExtension.substring(0, filenameWithExtension.lastIndexOf('.'));
            items.add(captchaFilename);
        }
    }

    private Artifact.Item getArtifactItem() {
        return artifact.getFiles().get("data");
    }

    private String getUsagePath() {
        String prefix = "captchaImage/";
        switch (usage) {
            case TRAIN:
                return prefix + "train";
            case TEST:
                return prefix + "test";
            case VALIDATION:
                return prefix + "validate";
            default:
                throw new IllegalArgumentException("Invalid usage");
        }
    }

    /** A builder for a {@link ai.djl.basicdataset.CaptchaDataset}. */
    public static final class Builder extends BaseBuilder<Builder> {

        private Repository repository;
        private Artifact artifact;
        private Usage usage;

        /** Constructs a new builder. */
        public Builder() {
            repository = BasicDatasets.REPOSITORY;
            usage = Usage.TRAIN;
            pipeline = new Pipeline(new ToTensor());
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
         * @return this builder
         */
        public Builder optRepository(Repository repository) {
            this.repository = repository;
            return this;
        }

        /**
         * Sets the optional artifact.
         *
         * @param artifact the artifact
         * @return this builder
         */
        public Builder optArtifact(Artifact artifact) {
            this.artifact = artifact;
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
            return this;
        }

        /**
         * Builds the {@link CaptchaDataset}.
         *
         * @return the {@link CaptchaDataset}
         */
        public CaptchaDataset build() {
            return new CaptchaDataset(this);
        }
    }
}
