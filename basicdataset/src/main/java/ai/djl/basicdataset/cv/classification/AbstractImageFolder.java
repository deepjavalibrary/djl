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

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A dataset for loading image files stored in a folder structure. */
public abstract class AbstractImageFolder extends ImageClassificationDataset {

    private static final Logger logger = LoggerFactory.getLogger(AbstractImageFolder.class);

    private static final Set<String> EXT =
            new HashSet<>(Arrays.asList(".jpg", ".jpeg", ".png", ".bmp", ".wbmp", ".gif"));

    protected List<String> synset;
    protected PairList<String, Integer> items;
    protected MRL mrl;
    protected boolean prepared;

    private int maxDepth;
    private Integer imageWidth;
    private Integer imageHeight;

    protected AbstractImageFolder(ImageFolderBuilder<?> builder) {
        super(builder);
        this.maxDepth = builder.maxDepth;
        this.imageWidth = builder.imageWidth;
        this.imageHeight = builder.imageHeight;
        this.synset = new ArrayList<>();
        this.items = new PairList<>();
        String path = builder.repository.getBaseUri().toString();
        mrl = MRL.undefined(builder.repository, DefaultModelZoo.GROUP_ID, path);
    }

    /** {@inheritDoc} */
    @Override
    protected Image getImage(long index) throws IOException {
        ImageFactory imageFactory = ImageFactory.getInstance();
        Pair<String, Integer> item = items.get(Math.toIntExact(index));
        Path imagePath = getImagePath(item.getKey());
        return imageFactory.fromFile(imagePath);
    }

    /** {@inheritDoc} */
    @Override
    protected long getClassNumber(long index) {
        Pair<String, Integer> item = items.get(Math.toIntExact(index));
        return item.getValue();
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return items.size();
    }

    /**
     * Returns the synsets of the ImageFolder dataset.
     *
     * @return a list that contains synsets
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public List<String> getSynset() throws IOException, TranslateException {
        prepare();
        return synset;
    }

    protected void listImages(Path root, List<String> classes) {
        int label = 0;
        for (String className : classes) {
            Path classFolder = root.resolve(className);
            if (!Files.isDirectory(classFolder)) {
                continue;
            }
            try (Stream<Path> stream = Files.walk(classFolder, maxDepth)) {
                final int classLabel = label;
                stream.forEach(
                        p -> {
                            if (isImage(p.toFile())) {
                                String path = p.toAbsolutePath().toString();
                                items.add(new Pair<>(path, classLabel));
                            }
                        });
            } catch (IOException e) {
                logger.warn("Failed to list images", e);
            }
            logger.debug("Loaded {} images in {}, class: {}", items.size(), classFolder, label);
            ++label;
        }
    }

    protected abstract Path getImagePath(String key);

    private boolean isImage(File file) {
        String path = file.getName();
        if (!file.isFile() || file.isHidden() || path.startsWith(".")) {
            return false;
        }

        int extensionIndex = path.lastIndexOf('.');
        if (extensionIndex < 0) {
            return false;
        }
        return EXT.contains(path.substring(extensionIndex).toLowerCase());
    }

    /** {@inheritDoc} */
    @Override
    public Optional<Integer> getImageWidth() {
        return Optional.ofNullable(imageWidth);
    }

    /** {@inheritDoc} */
    @Override
    public Optional<Integer> getImageHeight() {
        return Optional.ofNullable(imageWidth);
    }

    /** {@inheritDoc} */
    @Override
    public List<String> getClasses() {
        return synset;
    }

    /**
     * Used to build an {@link AbstractImageFolder}.
     *
     * @param <T> the builder type
     */
    public abstract static class ImageFolderBuilder<T extends ImageFolderBuilder<T>>
            extends BaseBuilder<T> {

        Repository repository;
        int maxDepth;
        Integer imageWidth;
        Integer imageHeight;

        protected ImageFolderBuilder() {
            maxDepth = 1;
        }

        /**
         * Sets the repository containing the image folder.
         *
         * @param repository the repository containing the image folder
         * @return this builder
         */
        public T setRepository(Repository repository) {
            this.repository = repository;
            return self();
        }

        /**
         * Sets the repository file path containing the image folder.
         *
         * @param path the repository file path containing the image folder
         * @return this builder
         */
        public T setRepositoryPath(String path) {
            this.repository = Repository.newInstance("images", path);
            return self();
        }

        /**
         * Sets the repository file path containing the image folder.
         *
         * @param path the repository file path containing the image folder
         * @return this builder
         */
        public T setRepositoryPath(Path path) {
            this.repository = Repository.newInstance("images", path);
            return self();
        }

        /**
         * Sets the depth of the image folder.
         *
         * @param maxDepth the maximum number of directory levels to visit
         * @return this builder
         */
        public T optMaxDepth(int maxDepth) {
            this.maxDepth = maxDepth;
            return self();
        }

        /**
         * Sets the size of the images.
         *
         * @param size the size (both width and height)
         * @return this builder
         */
        public T optImageSize(int size) {
            this.imageWidth = size;
            this.imageHeight = size;
            return self();
        }

        /**
         * Sets the width of the images.
         *
         * @param width the width of the images
         * @return this builder
         */
        public T optImageWidth(int width) {
            this.imageWidth = width;
            return self();
        }

        /**
         * Sets the height of the images.
         *
         * @param height the height of the images
         * @return this builder
         */
        public T optImageHeight(int height) {
            this.imageHeight = height;
            return self();
        }
    }
}
