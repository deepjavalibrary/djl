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

import ai.djl.Application.CV;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.Resource;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.Pipeline;
import ai.djl.util.Progress;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Coco image detection dataset from http://cocodataset.org/#home.
 *
 * <p>Each image might have different {@link ai.djl.ndarray.types.Shape}s.
 */
public class CocoDetection extends RandomAccessDataset {

    private static final String ARTIFACT_ID = "coco";

    private Usage usage;
    private Image.Flag flag;
    private List<Path> imagePaths;
    private List<double[][]> labels;

    private Resource resource;
    private boolean prepared;

    CocoDetection(Builder builder) {
        super(builder);
        usage = builder.usage;
        flag = builder.flag;
        imagePaths = new ArrayList<>();
        labels = new ArrayList<>();
        MRL mrl = MRL.dataset(CV.IMAGE_CLASSIFICATION, builder.groupId, builder.artifactId);
        resource = new Resource(builder.repository, mrl, "1.0");
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
    protected Record get(NDManager manager, long index) throws IOException {
        int idx = Math.toIntExact(index);
        NDList d =
                new NDList(
                        ImageFactory.getInstance()
                                .fromFile(imagePaths.get(idx))
                                .toNDArray(manager, flag));
        NDList l = new NDList(manager.create(labels.get(idx)));
        return new Record(d, l);
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Artifact artifact = resource.getDefaultArtifact();
        resource.prepare(artifact, progress);
        Path root = resource.getRepository().getResourceDirectory(artifact);

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
        CocoUtils coco = new CocoUtils(jsonFile);
        coco.prepare();
        List<Long> imageIds = coco.getImageIds();
        for (long id : imageIds) {
            Path imagePath = root.resolve(coco.getRelativeImagePath(id));
            List<double[]> labelOfImageId = getLabels(coco, id);
            if (!labelOfImageId.isEmpty()) {
                imagePaths.add(imagePath);
                labels.add(labelOfImageId.toArray(new double[0][]));
            }
        }
        prepared = true;
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
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

    private List<double[]> getLabels(CocoUtils coco, long imageId) {
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
    public static final class Builder extends BaseBuilder<Builder> {

        Image.Flag flag;
        Repository repository;
        String groupId;
        String artifactId;
        Usage usage;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Usage.TRAIN;
            flag = Image.Flag.COLOR;
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
         * Sets the optional color mode flag.
         *
         * @param flag the color mode flag
         * @return this builder
         */
        public Builder optFlag(Image.Flag flag) {
            this.flag = flag;
            return self();
        }

        /**
         * Builds the new {@link CocoDetection}.
         *
         * @return the new {@link CocoDetection}
         */
        public CocoDetection build() {
            if (pipeline == null) {
                pipeline = new Pipeline(new ToTensor());
            }
            return new CocoDetection(this);
        }
    }
}
