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

import ai.djl.modality.cv.Rectangle;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.modality.cv.util.NDImageUtils.Flag;
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
import java.net.URI;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Coco image detection dataset from http://cocodataset.org/#home.
 *
 * <p>Each image might have different {@link ai.djl.ndarray.types.Shape}s.
 */
public class CocoDetection extends RandomAccessDataset implements ZooDataset {

    private static final String ARTIFACT_ID = "coco";

    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;
    private Flag flag;

    private CocoUtils coco;
    private List<Path> imagePaths;
    private List<double[][]> labels;

    CocoDetection(Builder builder) {
        super(builder);
        repository = builder.repository;
        artifact = builder.artifact;
        usage = builder.usage;
        flag = builder.flag;
        imagePaths = new ArrayList<>();
        labels = new ArrayList<>();
    }

    /**
     * Creates a builder to build a {@link CocoDetection}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
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
    public Record get(NDManager manager, long index) throws IOException {
        int idx = Math.toIntExact(index);
        NDList d =
                new NDList(BufferedImageUtils.readFileToArray(manager, imagePaths.get(idx), flag));
        NDList l = new NDList(manager.create(labels.get(idx)));
        return new Record(d, l);
    }

    /** {@inheritDoc} */
    @Override
    public void prepareData(Usage usage) throws IOException {
        Path cacheDir = repository.getCacheDirectory();
        URI resourceUri = artifact.getResourceUri();
        Path root = cacheDir.resolve(resourceUri.getPath());

        Path jsonFile;
        switch (usage) {
            case TRAIN:
                jsonFile = root.resolve("annotations").resolve("instances_train2017.json");
                break;
            case TEST:
                jsonFile = root.resolve("annotations").resolve("instances_val2017.json");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        coco = new CocoUtils(jsonFile);
        coco.prepare();
        List<Long> imageIds = coco.getImageIds();
        for (long id : imageIds) {
            Path imagePath = root.resolve(coco.getRelativeImagePath(id));
            List<double[]> labelOfImageId = getLabels(id);
            if (!labelOfImageId.isEmpty()) {
                imagePaths.add(imagePath);
                labels.add(labelOfImageId.toArray(new double[0][]));
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return imagePaths.size();
    }

    private double[] convertRecToList(Rectangle rect) {
        double[] list = new double[5];
        list[0] = rect.getX();
        list[1] = rect.getY();
        list[2] = rect.getWidth();
        list[3] = rect.getHeight();
        return list;
    }

    private List<double[]> getLabels(long imageId) {
        List<Long> annotationIds = coco.getAnnotationIdByImageId(imageId);
        if (annotationIds == null) {
            return Collections.emptyList();
        }

        List<double[]> label = new ArrayList<>();
        for (long annotationId : annotationIds) {
            CocoMetadata.Annotation annotation = coco.getAnnotationById(annotationId);
            Rectangle bBox = annotation.getBoundingBox();
            if (annotation.getArea() > 0) {
                double[] list = convertRecToList(bBox);
                // add the category label
                // map the original one to incremental index
                list[4] = coco.mapCategoryId(annotation.getCategoryId());
                label.add(list);
            }
        }
        return label;
    }

    /** A builder to construct a {@link CocoDetection}. */
    @SuppressWarnings("rawtypes")
    public static final class Builder extends BaseBuilder<Builder> {

        private Flag flag;
        private Repository repository;
        private Artifact artifact;
        private Usage usage;

        /** Constructs a new builder. */
        public Builder() {
            repository = BasicDatasets.REPOSITORY;
            usage = Usage.TRAIN;
            pipeline = new Pipeline(new ToTensor());
            flag = NDImageUtils.Flag.COLOR;
        }

        /** {@inheritDoc} */
        @Override
        public Builder self() {
            return this;
        }

        /**
         * Sets the optional usage.
         *
         * @param usage the new usage
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
         * Sets the optional artifact.
         *
         * @param artifact the artifact
         * @return this builder
         */
        public Builder optArtifact(Artifact artifact) {
            this.artifact = artifact;
            return self();
        }

        /**
         * Sets the optional color mode flag.
         *
         * @param flag the color mode flag
         * @return this builder
         */
        public Builder optFlag(Flag flag) {
            this.flag = flag;
            return self();
        }

        /**
         * Builds the new {@link CocoDetection}.
         *
         * @return the new {@link CocoDetection}
         */
        public CocoDetection build() {
            return new CocoDetection(this);
        }
    }
}
