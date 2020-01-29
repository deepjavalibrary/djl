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
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.modality.cv.util.NDImageUtils.Flag;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Repository;
import ai.djl.repository.dataset.PreparedDataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.Pipeline;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** A dataset for loading image files stored in a folder structure. */
public abstract class AbstractImageFolder extends RandomAccessDataset implements PreparedDataset {

    private static final Set<String> EXT =
            new HashSet<>(Arrays.asList(".jpg", ".jpeg", ".png", ".bmp", ".wbmp", ".gif"));

    protected Repository repository;
    protected Flag flag;
    protected List<String> synset;
    protected PairList<String, Integer> items;

    protected AbstractImageFolder(ImageFolderBuilder<?> builder) {
        super(builder);
        this.flag = builder.flag;
        this.repository = builder.repository;
        this.synset = new ArrayList<>();
        this.items = new PairList<>();
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) throws IOException {
        Pair<String, Integer> item = items.get(Math.toIntExact(index));

        Path imagePath = getImagePath(item.getKey());
        NDArray array = BufferedImageUtils.readFileToArray(manager, imagePath, flag);
        NDList d = new NDList(array);
        NDList l = new NDList(manager.create(item.getValue()));
        return new Record(d, l);
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return items.size();
    }

    /**
     * Returns the synsets of the ImageFolder dataset.
     *
     * @return a list that contains synsets
     */
    public List<String> getSynset() {
        return synset;
    }

    protected void listImages(File root, List<String> classes) {
        int label = 0;
        for (String className : classes) {
            File classFolder = new File(root, className);
            if (!classFolder.exists() || !classFolder.isDirectory()) {
                continue;
            }
            File[] files = classFolder.listFiles(this::isImage);
            if (files == null) {
                continue;
            }

            for (File file : files) {
                String path = file.getAbsolutePath();
                items.add(new Pair<>(path, label));
            }
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
        Flag flag;

        protected ImageFolderBuilder() {
            flag = NDImageUtils.Flag.COLOR;
            pipeline = new Pipeline(new ToTensor());
        }

        /**
         * Sets the optional color mode flag.
         *
         * @param flag the color mode flag
         * @return this builder
         */
        public T optFlag(Flag flag) {
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
    }
}
