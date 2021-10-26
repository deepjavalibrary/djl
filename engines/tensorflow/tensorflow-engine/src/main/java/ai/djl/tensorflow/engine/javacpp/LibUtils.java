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

package ai.djl.tensorflow.engine.javacpp;

import ai.djl.util.Platform;
import ai.djl.util.Utils;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Enumeration;
import java.util.List;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "jnitensorflow";
    private static final Pattern VERSION_PATTERN =
            Pattern.compile("(\\d+\\.\\d+\\.\\d+(-[a-z]+)?)(-SNAPSHOT)?(-\\d+)?");

    private LibUtils() {}

    public static void loadLibrary() {
        String libName = getLibName();
        if (libName != null) {
            logger.debug("Loading TensorFlow library from: {}", libName);
            String path = new File(libName).getParentFile().toString();
            System.setProperty("org.bytedeco.javacpp.platform.preloadpath", path);
            // workaround javacpp physical memory check bug
            System.setProperty("org.bytedeco.javacpp.maxBytes", "0");
            System.setProperty("org.bytedeco.javacpp.maxPhysicalBytes", "0");
        }
        // defer to tensorflow-core-api to handle loading native library.
    }

    public static String getLibName() {
        String libName = LibUtils.findOverrideLibrary();
        if (libName == null) {
            libName = LibUtils.findLibraryInClasspath();
            if (libName == null) {
                libName = LIB_NAME;
            }
        }
        return libName;
    }

    private static String findOverrideLibrary() {
        String libPath = System.getenv("TENSORFLOW_LIBRARY_PATH");
        if (libPath != null) {
            String libName = findLibraryInPath(libPath);
            if (libName != null) {
                return libName;
            }
        }

        libPath = System.getProperty("java.library.path");
        if (libPath != null) {
            return findLibraryInPath(libPath);
        }
        return null;
    }

    private static synchronized String findLibraryInClasspath() {
        // TensorFlow doesn't support native library override
        Enumeration<URL> urls;
        try {
            urls =
                    Thread.currentThread()
                            .getContextClassLoader()
                            .getResources("native/lib/tensorflow.properties");
        } catch (IOException e) {
            logger.warn("", e);
            return null;
        }

        // No native jars
        if (!urls.hasMoreElements()) {
            String preferredVersion;
            try (InputStream is =
                    LibUtils.class.getResourceAsStream("/tensorflow-engine.properties")) {
                Properties prop = new Properties();
                prop.load(is);
                preferredVersion = prop.getProperty("tensorflow_version");
            } catch (IOException e) {
                throw new IllegalStateException("tensorflow-engine.properties not found.", e);
            }
            Platform platform = Platform.fromSystem(preferredVersion);
            return downloadTensorFlow(platform);
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
                return downloadTensorFlow(placeholder);
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read Tensorflow native library jar properties", e);
        }

        throw new IllegalStateException(
                "Your Tensorflow native library jar does not match your operating system. Make sure that the Maven Dependency Classifier matches your system type.");
    }

    private static String loadLibraryFromClasspath(Platform platform) {
        Path tmp = null;
        try {
            String libName = System.mapLibraryName(LIB_NAME);
            Path cacheFolder = Utils.getEngineCacheDir("tensorflow");
            String version = platform.getVersion();
            String flavor = platform.getFlavor();
            String classifier = platform.getClassifier();
            Path dir = cacheFolder.resolve(version + '-' + flavor + '-' + classifier);
            logger.debug("Using cache dir: {}", dir);

            Path path = dir.resolve(libName);
            if (Files.exists(path)) {
                return path.toAbsolutePath().toString();
            }
            Files.createDirectories(cacheFolder);
            tmp = Files.createTempDirectory(cacheFolder, "tmp");
            for (String file : platform.getLibraries()) {
                String libPath = "/native/lib/" + file;
                logger.info("Extracting {} to cache ...", libPath);
                try (InputStream is = LibUtils.class.getResourceAsStream(libPath)) {
                    if (is == null) {
                        throw new IllegalStateException("Tensorflow library not found: " + libPath);
                    }
                    Files.copy(is, tmp.resolve(file), StandardCopyOption.REPLACE_EXISTING);
                }
            }

            Utils.moveQuietly(tmp, dir);
            return path.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract Tensorflow native library", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static String findLibraryInPath(String libPath) {
        String[] paths = libPath.split(File.pathSeparator);
        String mapLibraryName = System.mapLibraryName(LIB_NAME);

        for (String path : paths) {
            File p = new File(path);
            if (!p.exists()) {
                continue;
            }
            if (p.isFile() && p.getName().endsWith(mapLibraryName)) {
                return p.getAbsolutePath();
            }

            File file = new File(path, mapLibraryName);
            if (file.exists() && file.isFile()) {
                return file.getAbsolutePath();
            }
        }
        return null;
    }

    private static String downloadTensorFlow(Platform platform) {
        String version = platform.getVersion();
        String os = platform.getOsPrefix();
        String classifier = platform.getClassifier();
        String cudaArch = platform.getCudaArch();
        String flavor = platform.getFlavor();

        String libName = System.mapLibraryName(LIB_NAME);
        Path cacheDir = Utils.getEngineCacheDir("tensorflow");
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

        Path tmp = null;
        String link = "https://publish.djl.ai/tensorflow-" + matcher.group(1);
        try (InputStream is = new URL(link + "/files.txt").openStream()) {
            Files.createDirectories(cacheDir);
            tmp = Files.createTempDirectory(cacheDir, "tmp");

            List<String> lines = Utils.readLines(is);
            boolean found = downloadFiles(lines, link, os, flavor, tmp);
            if (!found && cudaArch != null) {
                // fallback to cpu
                flavor = "cpu";
                dir = cacheDir.resolve(version + '-' + flavor + '-' + classifier);
                path = dir.resolve(libName);
                if (Files.exists(path)) {
                    logger.warn(
                            "No matching CUDA flavor for {} found: {}/sm_{}, fallback to CPU.",
                            os,
                            flavor,
                            cudaArch);
                    return path.toAbsolutePath().toString();
                }
                found = downloadFiles(lines, link, os, flavor, tmp);
            }

            if (!found) {
                throw new UnsupportedOperationException(
                        "TensorFlow engine does not support this platform: " + os);
            }

            Utils.moveQuietly(tmp, dir);
            return path.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to download Tensorflow native library", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static boolean downloadFiles(
            List<String> lines, String link, String os, String flavor, Path tmp)
            throws IOException {
        boolean found = false;
        for (String line : lines) {
            if (line.startsWith(os + '/' + flavor + '/')) {
                found = true;
                URL url = new URL(link + '/' + line.replace("+", "%2B"));
                String fileName = line.substring(line.lastIndexOf('/') + 1, line.length() - 3);
                logger.info("Downloading {} ...", url);
                try (InputStream fis = new GZIPInputStream(url.openStream())) {
                    Files.copy(fis, tmp.resolve(fileName), StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
        return found;
    }
}
