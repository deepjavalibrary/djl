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

import ai.djl.ndarray.NDArray;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * {@code ImageFactory} contains image creation mechanism on top of different platforms like PC and
 * Android. System will choose appropriate Factory based on the supported image type.
 */
public abstract class ImageFactory {

    private static ImageFactory factory = newInstance();

    private static ImageFactory newInstance() {
        String className = "ai.djl.modality.cv.BufferedImageFactory";
        if ("http://www.android.com/".equals(System.getProperty("java.vendor.url"))) {
            className = "ai.djl.android.core.BitmapImageFactory";
        }
        try {
            Class<? extends ImageFactory> clazz =
                    Class.forName(className).asSubclass(ImageFactory.class);
            return clazz.getConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Create new ImageFactory failed!", e);
        }
    }

    /**
     * Gets new instance of Image factory from the provided factory implementation.
     *
     * @return {@link ImageFactory}
     */
    public static ImageFactory getInstance() {
        return factory;
    }

    /**
     * Sets a custom instance of {@code ImageFactory}.
     *
     * @param factory a custom instance of {@code ImageFactory}
     */
    public static void setImageFactory(ImageFactory factory) {
        ImageFactory.factory = factory;
    }

    /**
     * Gets {@link Image} from file.
     *
     * @param path the path to the image
     * @return {@link Image}
     * @throws IOException Image not found or not readable
     */
    public abstract Image fromFile(Path path) throws IOException;

    /**
     * Gets {@link Image} from URL.
     *
     * @param url the URL to load from
     * @return {@link Image}
     * @throws IOException URL is not valid.
     */
    public abstract Image fromUrl(URL url) throws IOException;

    /**
     * Gets {@link Image} from URL.
     *
     * @param url the String represent URL to load from
     * @return {@link Image}
     * @throws IOException URL is not valid.
     */
    public Image fromUrl(String url) throws IOException {
        URI uri = URI.create(url);
        if (uri.isAbsolute()) {
            return fromUrl(uri.toURL());
        }
        return fromFile(Paths.get(url));
    }

    /**
     * Gets {@link Image} from {@link InputStream}.
     *
     * @param is {@link InputStream}
     * @return {@link Image}
     * @throws IOException image cannot be read from input stream.
     */
    public abstract Image fromInputStream(InputStream is) throws IOException;

    /**
     * Gets {@link Image} from varies Java image types.
     *
     * <p>Image can be BufferedImage or BitMap depends on platform
     *
     * @param image the image object.
     * @return {@link Image}
     */
    public abstract Image fromImage(Object image);

    /**
     * Gets {@link Image} from {@link NDArray}.
     *
     * @param array the NDArray with CHW format
     * @return {@link Image}
     */
    public abstract Image fromNDArray(NDArray array);
}
