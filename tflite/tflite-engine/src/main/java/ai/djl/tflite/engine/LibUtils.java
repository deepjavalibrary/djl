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
package ai.djl.tflite.engine;

import ai.djl.util.Platform;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Enumeration;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utilities for finding the TFLite Engine binary on the System.
 *
 * <p>The Engine will be searched for in a variety of locations in the following order:
 *
 * <ol>
 *   <li>In the path specified by the TFLite_LIBRARY_PATH environment variable
 * </ol>
 */
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "tensorflowlite_jni";

    private static final Pattern VERSION_PATTERN =
            Pattern.compile("(\\d+\\.\\d+\\.\\d+(-[a-z]+)?)(-SNAPSHOT)?(-\\d+)?");

    private LibUtils() {}

    public static void loadLibrary() {
        String libName = findLibraryInClasspath();
        if (libName == null) {
            System.loadLibrary(LIB_NAME); // NOPMD
            return;
        }
        logger.debug("Loading TFLite native library from: {}", libName);
        System.load(libName); // NOPMD
    }

    private static synchronized String findLibraryInClasspath() {
        Enumeration<URL> urls;
        try {
            urls =
                    Thread.currentThread()
                            .getContextClassLoader()
                            .getResources("native/lib/tflite.properties");
        } catch (IOException e) {
            logger.warn("", e);
            return null;
        }

        // No native jars
        if (!urls.hasMoreElements()) {
            logger.debug("tflite.properties not found in class path.");
            return null;
        }

        Platform systemPlatform = Platform.fromSystem();
        try {
            Platform matching = null;
            Platform placeholder = null;
            while (urls.hasMoreElements()) {
                URL url = urls.nextElement();
                Platform platform = Platform.fromUrl(url);
                if (platform.isPlaceholder()) {
                    placeholder = platform;
                } else if (platform.matches(systemPlatform)) {
                    matching = platform;
                    break;
                }
            }

            if (matching != null) {
                return loadLibraryFromClasspath(matching);
            }

            if (placeholder != null) {
                try {
                    return downloadTfLite(placeholder);
                } catch (IOException e) {
                    throw new IllegalStateException("Failed to download TFLite native library", e);
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read TFLite native library jar properties", e);
        }

        throw new IllegalStateException(
                "Your TFLite native library jar does not match your operating system. Make sure that the Maven Dependency Classifier matches your system type.");
    }

    private static String loadLibraryFromClasspath(Platform platform) {
        Path tmp = null;
        try {
            String libName = System.mapLibraryName(LIB_NAME);
            Path cacheFolder = getCacheDir();
            logger.debug("Using cache dir: {}", cacheFolder);

            Path dir = cacheFolder.resolve(platform.getVersion() + platform.getClassifier());
            Path path = dir.resolve(libName);
            if (Files.exists(path)) {
                return path.toAbsolutePath().toString();
            }
            Files.createDirectories(cacheFolder);
            tmp = Files.createTempDirectory(cacheFolder, "tmp");
            for (String file : platform.getLibraries()) {
                String libPath = "/native/lib/" + file;
                try (InputStream is = LibUtils.class.getResourceAsStream(libPath)) {
                    logger.info("Extracting {} to cache ...", file);
                    Files.copy(is, tmp.resolve(file), StandardCopyOption.REPLACE_EXISTING);
                }
            }

            Utils.moveQuietly(tmp, dir);
            return path.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract MXNet native library", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static String downloadTfLite(Platform platform) throws IOException {
        String version = platform.getVersion();
        String flavor = platform.getFlavor();
        if (flavor.isEmpty()) {
            flavor = "cpu";
        }
        String classifier = platform.getClassifier();
        String os = platform.getOsPrefix();

        String libName = System.mapLibraryName(LIB_NAME);
        Path cacheDir = getCacheDir();
        logger.debug("Using cache dir: {}", cacheDir);
        Path dir = cacheDir.resolve(version + '-' + flavor + '-' + classifier);
        Path path = dir.resolve(libName);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }

        Matcher matcher = VERSION_PATTERN.matcher(version);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Unexpected version: " + version);
        }
        String link = "https://publish.djl.ai/tflite-" + matcher.group(1);

        Files.createDirectories(cacheDir);
        Path tmp = null;
        try (InputStream is = new URL(link + "/files.txt").openStream()) {
            List<String> lines = Utils.readLines(is);
            if (flavor.startsWith("cu")
                    && !lines.contains(flavor + '/' + classifier + '/' + libName + ".gz")) {
                logger.warn("No matching cuda flavor for {} found: {}.", os, flavor);
                // fallback to CPU
                flavor = "cpu";

                // check again
                dir = cacheDir.resolve(version + '-' + flavor + '-' + classifier);
                path = dir.resolve(libName);
                if (Files.exists(path)) {
                    return path.toAbsolutePath().toString();
                }
            }

            tmp = Files.createTempDirectory(cacheDir, "tmp");
            for (String line : lines) {
                if (line.startsWith(flavor + '/' + classifier + '/')) {
                    URL url = new URL(link + '/' + line);
                    String fileName = line.substring(line.lastIndexOf('/') + 1, line.length() - 3);
                    logger.info("Downloading {} ...", url);
                    try (InputStream fis = new GZIPInputStream(url.openStream())) {
                        Files.copy(fis, tmp.resolve(fileName), StandardCopyOption.REPLACE_EXISTING);
                    }
                }
            }

            Utils.moveQuietly(tmp, dir);
            return path.toAbsolutePath().toString();
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static Path getCacheDir() {
        String cacheDir = System.getProperty("ENGINE_CACHE_DIR");
        if (cacheDir == null || cacheDir.isEmpty()) {
            cacheDir = System.getenv("ENGINE_CACHE_DIR");
            if (cacheDir == null || cacheDir.isEmpty()) {
                cacheDir = System.getProperty("DJL_CACHE_DIR");
                if (cacheDir == null || cacheDir.isEmpty()) {
                    cacheDir = System.getenv("DJL_CACHE_DIR");
                    if (cacheDir == null || cacheDir.isEmpty()) {
                        String userHome = System.getProperty("user.home");
                        return Paths.get(userHome, ".djl.ai").resolve("tflite");
                    }
                }
            }
        }
        return Paths.get(cacheDir, "tflite");
    }
}
