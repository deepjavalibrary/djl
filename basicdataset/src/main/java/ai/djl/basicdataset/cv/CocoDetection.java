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
package ai.djl.basicdataset.cv;

import ai.djl.Application;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.translate.Pipeline;
import ai.djl.util.PairList;
import ai.djl.util.Progress;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Coco image detection dataset from http://cocodataset.org/#home.
 *
 * <p>Each image might have different {@link ai.djl.ndarray.types.Shape}s.
 */
public class CocoDetection extends ObjectDetectionDataset {

    private static final String ARTIFACT_ID = "coco";
    private static final String VERSION = "1.0";

    private Usage usage;
    private List<Path> imagePaths;
    private List<PairList<Long, Rectangle>> labels;

    private MRL mrl;
    private boolean prepared;

    CocoDetection(Builder builder) {
        super(builder);
        usage = builder.usage;
        mrl = builder.getMrl();
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

    @Override
    public PairList<Long, Rectangle> getObjects(long index) {
        return labels.get(Math.toIntExact(index));
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);
        Path root = mrl.getRepository().getResourceDirectory(artifact);

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
            PairList<Long, Rectangle> labelOfImageId = getLabels(coco, id);
            if (!labelOfImageId.isEmpty()) {
                imagePaths.add(imagePath);
                labels.add(labelOfImageId);
            }
        }
        prepared = true;
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return imagePaths.size();
    }

    private PairList<Long, Rectangle> getLabels(CocoUtils coco, long imageId) {
        List<Long> annotationIds = coco.getAnnotationIdByImageId(imageId);
        if (annotationIds == null) {
            return new PairList<>();
        }

        PairList<Long, Rectangle> label = new PairList<>(annotationIds.size());
        for (long annotationId : annotationIds) {
            CocoMetadata.Annotation annotation = coco.getAnnotationById(annotationId);
            if (annotation.getArea() > 0) {
                double[] box = annotation.getBoundingBox();
                long labelClass = coco.mapCategoryId(annotation.getCategoryId());
                Rectangle objectLocation = new Rectangle(new Point(box[0], box[1]), box[2], box[3]);
                label.add(labelClass, objectLocation);
            }
        }
        return label;
    }

    @Override
    protected Image getImage(long index) throws IOException {
        int idx = Math.toIntExact(index);
        return ImageFactory.getInstance().fromFile(imagePaths.get(idx));
    }

    @Override
    public Optional<Integer> getImageWidth() {
        return Optional.empty();
    }

    @Override
    public Optional<Integer> getImageHeight() {
        return Optional.empty();
    }

    /** A builder to construct a {@link CocoDetection}. */
    public static final class Builder extends ImageDataset.BaseBuilder<Builder> {

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

        MRL getMrl() {
            return repository.dataset(Application.CV.ANY, groupId, artifactId, VERSION);
        }
    }
}
