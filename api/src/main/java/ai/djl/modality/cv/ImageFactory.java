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
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * {@code ImageFactory} contains image creation mechanism on top of different platforms like PC and
 * Android. System will choose appropriate Factory based on the supported image type.
 */
public abstract class ImageFactory {

    private static final Logger logger = LoggerFactory.getLogger(ImageFactory.class);

    private static final String[] FACTORIES = {
        "ai.djl.opencv.OpenCVImageFactory",
        "ai.djl.modality.cv.BufferedImageFactory",
        "ai.djl.android.core.BitmapImageFactory"
    };

    private static final Pattern URL_PATTERN = Pattern.compile("^data:image/\\w+;base64,(.+)");

    private static ImageFactory factory = newInstance();

    private static ImageFactory newInstance() {
        int index = 0;
        if ("http://www.android.com/".equals(System.getProperty("java.vendor.url"))) {
            index = 2;
        }
        for (int i = index; i < FACTORIES.length; ++i) {
            try {
                Class<? extends ImageFactory> clazz =
                        Class.forName(FACTORIES[i]).asSubclass(ImageFactory.class);
                return clazz.getConstructor().newInstance();
            } catch (ReflectiveOperationException | ExceptionInInitializerError e) {
                logger.trace("", e);
            }
        }
        throw new IllegalStateException("Create new ImageFactory failed!");
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
    public Image fromUrl(URL url) throws IOException {
        try (InputStream is = url.openStream()) {
            return fromInputStream(is);
        }
    }

    /**
     * Gets {@link Image} from string representation.
     *
     * <p>A string that is not an absolute URL is loaded as a local file. Use {@link
     * #fromRequestUrl(String)} for a URL that arrived with a request.
     *
     * @param url the String represent URL or base64 encoded image to load from
     * @return {@link Image}
     * @throws IOException URL is not valid.
     */
    public Image fromUrl(String url) throws IOException {
        Matcher m = URL_PATTERN.matcher(url);
        if (m.matches()) {
            // url="data:image/png;base64,..."
            byte[] buf = Base64.getDecoder().decode(m.group(1));
            try (InputStream is = new ByteArrayInputStream(buf)) {
                return fromInputStream(is);
            }
        }

        URI uri = URI.create(url);
        if (uri.isAbsolute()) {
            return fromUrl(uri.toURL());
        }
        return fromFile(Paths.get(url));
    }

    /**
     * Gets {@link Image} from a URL that arrived with a request, such as the {@code image_url} of a
     * JSON input.
     *
     * <p>Unlike {@link #fromUrl(String)}, this only accepts a base64 {@code data:} URI or an {@code
     * http} or {@code https} URL, and the URL is fetched with {@link Utils#openUrl(URL)}, so it
     * gets the same checks as other remote resources. Set {@code DJL_ALLOW_INSECURE_URL=true} (or
     * {@code -Dai.djl.allow_insecure_url=true}) to accept any URL or file path, as {@link
     * #fromUrl(String)} does; as for other remote resources, nothing is fetched in offline mode.
     *
     * @param url the {@code data:} URI, or the {@code http} or {@code https} URL
     * @return {@link Image}
     * @throws IOException if the URL is not supported, or the image cannot be read
     */
    public Image fromRequestUrl(String url) throws IOException {
        if (URL_PATTERN.matcher(url).matches()) {
            return fromUrl(url);
        }
        URI uri = URI.create(url);
        if (Utils.isInsecureUrlAllowed()) {
            if (!uri.isAbsolute()) {
                return fromFile(Paths.get(url));
            }
        } else {
            // The URL comes from whoever sent the request, so it is limited to a remote fetch,
            // which Utils.openUrl checks. A local file path or a file: URL is not accepted here.
            String scheme = uri.getScheme();
            if (!"http".equalsIgnoreCase(scheme) && !"https".equalsIgnoreCase(scheme)) {
                throw new IOException(
                        "Unsupported image URL: only data:, http and https URLs are accepted. Set"
                                + " DJL_ALLOW_INSECURE_URL=true to override.");
            }
        }
        // Utils.openUrl applies offline mode even when DJL_ALLOW_INSECURE_URL is set.
        try (InputStream is = Utils.openUrl(uri.toURL())) {
            return fromInputStream(is);
        }
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

    /**
     * Gets {@link Image} from array.
     *
     * @param pixels the array of ARGB values used to initialize the pixels.
     * @param width the width of the image
     * @param height the height of the image
     * @return {@link Image}
     */
    public abstract Image fromPixels(int[] pixels, int width, int height);
}
