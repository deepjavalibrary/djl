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

import ai.djl.util.Progress;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * A dataset for loading image files stored in a folder structure.
 *
 * <pre>
 *  The image folder should be structured as follows:
 *       root/shoes/Aerobic Shoes1.png
 *       root/shoes/Aerobic Shose2.png
 *       ...
 *       root/boots/Black Boots.png
 *       root/boots/White Boots.png
 *       ...
 *       root/pumps/Red Pumps
 *       root/pumps/Pink Pumps
 *       ...
 *  here shoes, boots, pumps are your labels
 *  </pre>
 */
public final class ImageFolder extends AbstractImageFolder {

    private boolean prepared;

    private ImageFolder(ImageFolderBuilder<?> builder) {
        super(builder);
        loadSynset();
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
    public void prepare(Progress progress) {
        if (!prepared) {
            File root = new File(repository.getBaseUri());
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
        File root = new File(repository.getBaseUri());
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
            return new ImageFolder(this);
        }
    }
}
