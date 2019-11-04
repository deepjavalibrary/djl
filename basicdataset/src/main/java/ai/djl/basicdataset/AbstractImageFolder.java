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

import ai.djl.basicdataset.utils.ThrowingFunction;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.modality.cv.util.NDImageUtils.Flag;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A dataset for loading image files stored in a folder structure. */
public abstract class AbstractImageFolder extends RandomAccessDataset implements PreparedDataset {

    private static final Set<String> EXT =
            new HashSet<>(Arrays.asList(".jpg", ".jpeg", ".png", ".bmp", ".wbmp", ".gif"));
    private static final Logger logger = LoggerFactory.getLogger(AbstractImageFolder.class);

    protected Repository repository;
    protected Flag flag;
    protected List<String> synsets;
    protected PairList<String, Integer> items;

    public AbstractImageFolder(ImageFolderBuilder<?> builder) {
        super(builder);
        this.flag = builder.flag;
        this.repository = builder.repository;
        if (pipeline == null) {
            pipeline = new Pipeline();
            pipeline.add(new ToTensor());
        }
        this.synsets = new ArrayList<>();
        this.items = new PairList<>();
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) throws IOException {
        Pair<String, Integer> item = items.get(Math.toIntExact(index));

        Path imagePath = getImagePath(item.getKey());
        NDArray array = BufferedImageUtils.readFileToArray(manager, imagePath);
        NDList d = new NDList(array);
        NDList l = new NDList(manager.create(item.getValue()));
        return new Record(d, l);
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return items.size();
    }

    protected void listImages(String root) throws IOException {
        File[] dir = new File(root).listFiles(f -> f.isDirectory() && !f.getName().startsWith("."));
        if (dir == null || dir.length == 0) {
            throw new IllegalArgumentException(root + " not found or didn't have any file in it");
        }
        Arrays.sort(dir);
        String[] classes = new String[dir.length];
        for (int i = 0; i < dir.length; i++) {
            classes[i] = dir[i].getName();
        }
        listImages(root, classes);
    }

    protected void listImages(String root, String[] classes) throws IOException {
        ThrowingFunction<String, String[], IOException> listClass =
                className -> {
                    File classFolder = new File(root + "/" + className);
                    if (!classFolder.exists() || !classFolder.isDirectory()) {
                        return new String[0];
                    }
                    File[] files =
                            classFolder.listFiles(
                                    f ->
                                            !f.isDirectory()
                                                    && !f.isHidden()
                                                    && !f.getName().startsWith("."));
                    if (files == null) {
                        return new String[0];
                    }

                    String[] images = new String[files.length];
                    for (int i = 0; i < files.length; i++) {
                        images[i] = files[i].getAbsolutePath();
                    }
                    return images;
                };
        listImages(classes, listClass);
    }

    protected void listImages(Repository repository, Artifact.Item item) throws IOException {
        String[] classes = repository.listDirectory(item, "");
        if (classes.length == 0) {
            throw new IllegalArgumentException("No classes found in " + item.getName());
        }
        listImages(repository, item, classes);
    }

    protected void listImages(Repository repository, Artifact.Item item, String[] classes)
            throws IOException {
        ThrowingFunction<String, String[], IOException> listClass =
                className ->
                        Arrays.stream(repository.listDirectory(item, className))
                                .map(image -> className + "/" + image)
                                .toArray(String[]::new);
        listImages(classes, listClass);
    }

    private void listImages(
            String[] classes, ThrowingFunction<String, String[], IOException> listClass)
            throws IOException {
        for (int label = 0; label < classes.length; label++) {
            String className = classes[label];
            synsets.add(className);
            String[] images = listClass.apply(className);
            if (images == null || images.length == 0) {
                logger.warn("Bad class folder: {}", className);
                continue;
            }
            for (String image : images) {
                if (isImage(image)) {
                    items.add(new Pair<>(image, label));
                } else {
                    logger.warn("ImageIO didn't support {} Ignoring... ", image);
                }
            }
        }
    }

    protected abstract Path getImagePath(String key);

    private boolean isImage(String path) {
        int extensionIndex = path.lastIndexOf('.');
        if (extensionIndex < 0) {
            return false;
        }
        return EXT.contains(path.substring(extensionIndex).toLowerCase());
    }

    @SuppressWarnings("rawtypes")
    public abstract static class ImageFolderBuilder<T extends ImageFolderBuilder>
            extends BaseBuilder<T> {
        private Repository repository;
        Flag flag = NDImageUtils.Flag.COLOR;

        public T optFlag(Flag flag) {
            this.flag = flag;
            return self();
        }

        public T setRepository(Repository repository) {
            this.repository = repository;
            return self();
        }
    }
}
