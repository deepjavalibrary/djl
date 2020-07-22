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

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Repository;
import ai.djl.repository.Resource;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
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
import java.util.Set;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A dataset for loading image files stored in a folder structure. */
public abstract class AbstractImageFolder extends RandomAccessDataset {

    private static final Logger logger = LoggerFactory.getLogger(AbstractImageFolder.class);

    private static final Set<String> EXT =
            new HashSet<>(Arrays.asList(".jpg", ".jpeg", ".png", ".bmp", ".wbmp", ".gif"));

    protected Image.Flag flag;
    protected List<String> synset;
    protected PairList<String, Integer> items;
    protected Resource resource;
    protected boolean prepared;

    private int maxDepth;

    protected AbstractImageFolder(ImageFolderBuilder<?> builder) {
        super(builder);
        this.flag = builder.flag;
        this.maxDepth = builder.maxDepth;
        this.synset = new ArrayList<>();
        this.items = new PairList<>();
        this.resource = new Resource(builder.repository, null, "1.0");
    }

    /** {@inheritDoc} */
    @Override
    protected Record get(NDManager manager, long index) throws IOException {
        Pair<String, Integer> item = items.get(Math.toIntExact(index));

        Path imagePath = getImagePath(item.getKey());
        NDArray array = ImageFactory.getInstance().fromFile(imagePath).toNDArray(manager, flag);
        NDList d = new NDList(array);
        NDList l = new NDList(manager.create(item.getValue()));
        return new Record(d, l);
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

    /**
     * Used to build an {@link AbstractImageFolder}.
     *
     * @param <T> the builder type
     */
    @SuppressWarnings("rawtypes")
    public abstract static class ImageFolderBuilder<T extends ImageFolderBuilder>
            extends BaseBuilder<T> {

        Repository repository;
        Image.Flag flag;
        int maxDepth;

        protected ImageFolderBuilder() {
            flag = Image.Flag.COLOR;
            maxDepth = 1;
        }

        /**
         * Sets the optional color mode flag.
         *
         * @param flag the color mode flag
         * @return this builder
         */
        public T optFlag(Image.Flag flag) {
            this.flag = flag;
            return self();
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
    }
}
