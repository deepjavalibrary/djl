/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;

/**
 * {@code ImageFactory} contains image creation mechanism on top of different platforms like PC and
 * Android. System will choose appropriate Factory based on the supported image type.
 */
public interface ImageFactory {

    /**
     * Get new instance of Image factory from the provided factory implementation.
     *
     * @return {@link ImageFactory}
     */
    static ImageFactory newInstance() {
        String className = "ai.djl.api.cv.BufferedImageFactory";
        if (System.getProperty("java.vendor.url").equals("http://www.android.com/")) {
            className = "ai.djl.android.cv.BitMapFactory";
        }
        try {
            Class<? extends ImageFactory> clazz =
                    Class.forName(className).asSubclass(ImageFactory.class);
            return clazz.newInstance();
        } catch (InstantiationException | IllegalAccessException | ClassNotFoundException e) {
            throw new IllegalStateException("Create new ImageFactory failed!", e);
        }
    }

    /**
     * Gets {@link Image} from file.
     *
     * @param path the path to the image
     * @return {@link Image}
     * @throws IOException Image not found or not readable
     */
    Image fromFile(Path path) throws IOException;

    /**
     * Gets {@link Image} from URL.
     *
     * @param url the String represent URL to load from
     * @return {@link Image}
     * @throws IOException URL is not valid.
     */
    Image fromUrl(String url) throws IOException;

    /**
     * Gets {@link Image} from {@link InputStream}.
     *
     * @param is {@link InputStream}
     * @return {@link Image}
     */
    Image fromInputStream(InputStream is);

    /**
     * Gets {@link Image} from varies Java image types.
     *
     * <p>Image can be BufferedImage or BitMap depends on platform
     *
     * @param image the image object.
     * @return {@link Image}
     */
    Image fromImage(Object image);
}
