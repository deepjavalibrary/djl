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

package ai.djl.onnxruntime.engine;

import ai.djl.util.Platform;
import ai.djl.util.Utils;
import java.io.File;
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
 * Utilities for finding the OnnxRuntime Engine binary on the System.
 *
 * <p>The Engine will be searched for in a variety of locations in the following order:
 *
 * <ol>
 *   <li>In the path specified by the ONNXRUNTIME_LIBRARY_PATH environment variable
 *   <li>In a jar file location in the classpath. These jars can be created with the
 *       onnxruntime-native module.
 * </ol>
 */
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {
    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);
    private static final Pattern VERSION_PATTERN =
            Pattern.compile("(\\d+\\.\\d+\\.\\d+(-\\w)?)(-SNAPSHOT)?(-\\d+)?");

    private static final String LIB_NAME = "onnxruntime4j_jni";
    private static final String NATIVE_LIB_NAME = "onnxruntime";

    private LibUtils() {}

    public static void prepareLibrary() {
        // get the directory of the file
        Path libPath = findOverrideLibrary();
        if (libPath == null) {
            libPath = findNativeLibrary();
        } else {
            libPath = libPath.getParent();
        }
        if (libPath == null) {
            throw new IllegalStateException("ONNX Runtime Library now found!");
        }

        String jniPath = libPath.resolve(System.mapLibraryName(LIB_NAME)).toString();
        String nativePath = libPath.resolve(System.mapLibraryName(NATIVE_LIB_NAME)).toString();
        System.setProperty("onnxruntime.native." + LIB_NAME + ".path", jniPath);
        System.setProperty("onnxruntime.native." + NATIVE_LIB_NAME + ".path", nativePath);
        logger.debug("Loading onnxruntime JNI from: {}", jniPath);
        logger.debug("Loading onnxruntime native library from: {}", nativePath);
    }

    private static Path findOverrideLibrary() {
        String libPath = System.getenv("ONNXRUNTIME_LIBRARY_PATH");
        if (libPath != null) {
            Path path = findLibraryInPath(libPath);
            if (path != null) {
                return path;
            }
        }

        libPath = System.getProperty("java.library.path");
        if (libPath != null) {
            return findLibraryInPath(libPath);
        }
        return null;
    }

    private static Path findLibraryInPath(String libPath) {
        String[] paths = libPath.split(File.pathSeparator);
        String mappedLibName = System.mapLibraryName(LIB_NAME);

        for (String path : paths) {
            File p = new File(path);
            if (!p.exists()) {
                continue;
            }
            if (p.isFile() && p.getName().endsWith(mappedLibName)) {
                return p.toPath();
            }

            File file = new File(path, mappedLibName);
            if (file.exists() && file.isFile()) {
                return file.toPath();
            }
        }
        return null;
    }

    private static synchronized Path findNativeLibrary() {
        Enumeration<URL> urls;
        try {
            urls =
                    Thread.currentThread()
                            .getContextClassLoader()
                            .getResources("native/lib/onnxruntime.properties");
        } catch (IOException e) {
            logger.warn("", e);
            return null;
        }
        // No native jars
        if (!urls.hasMoreElements()) {
            logger.debug("onnxruntime.properties not found in class path.");
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
                return copyNativeLibraryFromClasspath(matching);
            }

            if (placeholder != null) {
                try {
                    return downloadOnnxRuntime(placeholder);
                } catch (IOException e) {
                    throw new IllegalStateException(
                            "Failed to download ONNXRuntime native library", e);
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read ONNXRuntime native library jar properties", e);
        }
        throw new IllegalStateException(
                "Your ONNXRuntime native library jar does not match your operating system. Make sure the Maven Dependency Classifier matches your system type.");
    }

    private static Path copyNativeLibraryFromClasspath(Platform platform) {
        Path tmp = null;
        String version = platform.getVersion();
        String flavor = platform.getFlavor();
        String classifier = platform.getClassifier();
        try {
            String libName = System.mapLibraryName(NATIVE_LIB_NAME);
            Path cacheDir = getCacheDir();
            logger.debug("Using cache dir: {}", cacheDir);
            Path dir = cacheDir.resolve(version + flavor + '-' + classifier);
            Path path = dir.resolve(libName);
            if (Files.exists(path)) {
                return dir;
            }

            Files.createDirectories(cacheDir);
            tmp = Files.createTempDirectory(cacheDir, "tmp");
            for (String file : platform.getLibraries()) {
                String libPath = "/native/lib/" + file;
                try (InputStream is = LibUtils.class.getResourceAsStream(libPath)) {
                    Files.copy(is, tmp.resolve(file), StandardCopyOption.REPLACE_EXISTING);
                }
            }

            Utils.moveQuietly(tmp, dir);
            return dir;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract ONNXRuntime native library", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static Path downloadOnnxRuntime(Platform platform) throws IOException {
        String version = platform.getVersion();
        String flavor = platform.getFlavor();
        if (flavor.isEmpty()) {
            flavor = "cpu";
        }
        String classifier = platform.getClassifier();
        String os = platform.getOsPrefix();

        String libName = System.mapLibraryName(NATIVE_LIB_NAME);
        Path cacheDir = getCacheDir();
        logger.debug("Using cache dir: {}", cacheDir);
        Path dir = cacheDir.resolve(version + flavor + '-' + classifier);
        Path path = dir.resolve(libName);
        if (Files.exists(path)) {
            return dir;
        }
        // if files not found
        Files.createDirectories(cacheDir);
        Path tmp = Files.createTempDirectory(cacheDir, "tmp");

        Matcher matcher = VERSION_PATTERN.matcher(version);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Unexpected version: " + version);
        }
        String link = "https://djl-ai.s3.amazonaws.com/publish/onnxruntime-" + matcher.group(1);
        try (InputStream is = new URL(link + "/files.txt").openStream()) {
            List<String> lines = Utils.readLines(is);
            for (String line : lines) {
                if (line.startsWith(flavor + '/' + os + '/')) {
                    URL url = new URL(link + '/' + line);
                    String fileName = line.substring(line.lastIndexOf('/') + 1, line.length() - 3);
                    logger.info("Downloading {} ...", fileName);
                    try (InputStream fis = new GZIPInputStream(url.openStream())) {
                        Files.copy(fis, tmp.resolve(fileName), StandardCopyOption.REPLACE_EXISTING);
                    }
                }
            }

            Utils.moveQuietly(tmp, dir);
            return dir;
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
                        return Paths.get(userHome, ".djl.ai/onnxruntime");
                    }
                }
                return Paths.get(cacheDir, "onnxruntime");
            }
        }
        return Paths.get(cacheDir, ".djl.ai/onnxruntime");
    }
}
