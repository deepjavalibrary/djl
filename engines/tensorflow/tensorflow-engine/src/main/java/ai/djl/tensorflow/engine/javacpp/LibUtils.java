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

import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Platform;
import ai.djl.util.Utils;

import org.bytedeco.javacpp.Loader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

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

            // https://github.com/deepjavalibrary/djl/issues/2318
            Loader.loadProperties(true);
        }
        // defer to tensorflow-core-api to handle loading native library.
    }

    public static String getLibName() {
        String libName = LibUtils.findOverrideLibrary();
        if (libName == null) {
            libName = LibUtils.findLibraryInClasspath();
        }
        return libName;
    }

    private static String findOverrideLibrary() {
        String libPath = Utils.getEnvOrSystemProperty("TENSORFLOW_LIBRARY_PATH");
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
        Platform platform = Platform.detectPlatform("tensorflow");
        if (platform.isPlaceholder()) {
            return downloadTensorFlow(platform);
        }
        return loadLibraryFromClasspath(platform);
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
                String libPath = "native/lib/" + file;
                logger.info("Extracting {} to cache ...", libPath);
                try (InputStream is = ClassLoaderUtils.getResourceAsStream(libPath)) {
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
        String link = "https://publish.djl.ai/tensorflow/" + matcher.group(1);
        try (InputStream is = Utils.openUrl(link + "/files.txt")) {
            Files.createDirectories(cacheDir);
            tmp = Files.createTempDirectory(cacheDir, "tmp");

            List<String> lines = Utils.readLines(is);
            boolean found = downloadFiles(lines, link, classifier, flavor, tmp);
            if (!found && cudaArch != null) {
                // fallback to cpu
                String cpuFlavor = "cpu";
                dir = cacheDir.resolve(version + '-' + cpuFlavor + '-' + classifier);
                path = dir.resolve(libName);
                if (Files.exists(path)) {
                    logger.warn(
                            "No matching CUDA flavor for {} found: {}/sm_{}, fallback to CPU.",
                            classifier,
                            flavor,
                            cudaArch);
                    return path.toAbsolutePath().toString();
                }
                flavor = cpuFlavor;
                found = downloadFiles(lines, link, classifier, flavor, tmp);
            }

            if (!found) {
                throw new IllegalStateException(
                        "No TensorFlow native library matches your operating system: " + platform);
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
            List<String> lines, String link, String classifier, String flavor, Path tmp)
            throws IOException {
        boolean found = false;
        String prefix;
        if (flavor.startsWith("cu12") && "linux-x86_64".equals(classifier)) {
            prefix = "cu121/linux-x86_64";
        } else {
            prefix = flavor + '/' + classifier + '/';
        }
        for (String line : lines) {
            if (line.startsWith(prefix)) {
                found = true;
                URL url = new URL(link + '/' + line.replace("+", "%2B"));
                String fileName = line.substring(line.lastIndexOf('/') + 1, line.length() - 3);
                logger.info("Downloading {} ...", url);
                try (InputStream fis = new GZIPInputStream(Utils.openUrl(url))) {
                    Files.copy(fis, tmp.resolve(fileName), StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
        return found;
    }
}
