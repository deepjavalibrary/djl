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

import ai.djl.Application.CV;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.Pipeline;
import ai.djl.util.Progress;
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
public class CaptchaDataset extends RandomAccessDataset {

    private static final String ARTIFACT_ID = "captcha";
    private static final String VERSION = "1.1";

    public static final int IMAGE_WIDTH = 160;
    public static final int IMAGE_HEIGHT = 60;
    public static final int CAPTCHA_LENGTH = 6;
    public static final int CAPTCHA_OPTIONS = 11;

    private Usage usage;
    private List<String> items;
    private Artifact.Item dataItem;
    private String pathPrefix;

    private MRL mrl;
    private boolean prepared;

    /**
     * Creates a new instance of {@link CaptchaDataset}.
     *
     * @param builder a builder with the necessary configurations
     */
    public CaptchaDataset(Builder builder) {
        super(builder);
        this.usage = builder.usage;
        mrl = builder.getMrl();
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
        Path imagePath = mrl.getRepository().getFile(dataItem, pathPrefix + '/' + item + ".jpeg");
        NDArray imageArray =
                ImageFactory.getInstance()
                        .fromFile(imagePath)
                        .toNDArray(manager, Image.Flag.GRAYSCALE);
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
    protected long availableSize() {
        return items.size();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);

        dataItem = artifact.getFiles().get("data");
        pathPrefix = getUsagePath();
        items = new ArrayList<>();
        for (String filenameWithExtension :
                mrl.getRepository().listDirectory(dataItem, pathPrefix)) {
            String captchaFilename =
                    filenameWithExtension.substring(0, filenameWithExtension.lastIndexOf('.'));
            items.add(captchaFilename);
        }
        prepared = true;
    }

    private String getUsagePath() {
        switch (usage) {
            case TRAIN:
                return "train";
            case TEST:
                return "test";
            case VALIDATION:
                return "validate";
            default:
                throw new IllegalArgumentException("Invalid usage");
        }
    }

    /** A builder for a {@link CaptchaDataset}. */
    public static final class Builder extends BaseBuilder<Builder> {

        Repository repository;
        String groupId;
        String artifactId;
        Usage usage;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Dataset.Usage.TRAIN;
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

        MRL getMrl() {
            return repository.dataset(CV.ANY, groupId, artifactId, VERSION);
        }
    }
}
