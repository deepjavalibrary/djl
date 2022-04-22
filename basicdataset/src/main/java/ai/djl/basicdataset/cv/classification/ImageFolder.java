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

import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.translate.Pipeline;
import ai.djl.util.Progress;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * A dataset for loading image files stored in a folder structure.
 *
 * <p>Below is an example directory layout for the image folder:
 *
 * <pre>
 *  The image folder should be structured as follows:
 *       root/shoes/Aerobic Shoes1.png
 *       root/shoes/Aerobic Shose2.png
 *       ...
 *       root/boots/Black Boots.png
 *       root/boots/White Boots.png
 *       ...
 *       root/pumps/Red Pumps.png
 *       root/pumps/Pink Pumps.png
 *       ...
 *
 *  here shoes, boots, pumps are your labels
 *  </pre>
 *
 * <p>Here, the dataset will take the folder names (shoes, boots, bumps) in sorted order as your
 * labels. Nested folder structures are not currently supported.
 *
 * <p>Then, you can create your instance of the dataset as follows:
 *
 * <pre>
 * // set the image folder path
 * Repository repository = Repository.newInstance("folder", Paths.get("/path/to/imagefolder/root");
 * ImageFolder dataset =
 *     new ImageFolder.Builder()
 *         .setRepository(repository)
 *         .addTransform(new Resize(100, 100)) // Use image transforms as necessary for your data
 *         .addTransform(new ToTensor()) // Usually required as the last transform to convert images to tensors
 *         .setSampling(batchSize, true)
 *         .build();
 *
 * // call prepare before using
 * dataset.prepare();
 *
 * // to get the synset or label names
 * List&gt;String&lt; synset = dataset.getSynset();
 * </pre>
 */
public final class ImageFolder extends AbstractImageFolder {

    private ImageFolder(ImageFolderBuilder<?> builder) {
        super(builder);
    }

    /**
     * Creates a new builder to build a {@link ImageFolder}.
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
        if (!prepared) {
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

    /** A builder for the {@link ImageFolder}. */
    public static final class Builder extends ImageFolderBuilder<Builder> {

        Builder() {}

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds the {@link ImageFolder}.
         *
         * @return the {@link ImageFolder}
         */
        public ImageFolder build() {
            if (pipeline == null) {
                pipeline = new Pipeline(new ToTensor());
            }
            return new ImageFolder(this);
        }
    }
}
