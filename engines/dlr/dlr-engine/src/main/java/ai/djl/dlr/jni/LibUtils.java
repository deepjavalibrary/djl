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
package ai.djl.dlr.jni;

import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Platform;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Utilities for finding the DLR Engine binary on the System.
 *
 * <p>The Engine will be searched for in a variety of locations in the following order:
 *
 * <ol>
 *   <li>In the path specified by the DLR_LIBRARY_PATH environment variable
 * </ol>
 */
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "djl_dlr";
    private static final String NATIVE_LIB_NAME = "dlr";

    private static final Pattern VERSION_PATTERN =
            Pattern.compile("(\\d+\\.\\d+\\.\\d+(-[a-z]+)?)(-SNAPSHOT)?(-\\d+)?");

    private LibUtils() {}

    public static void loadLibrary() {
        String libName = findNativeOverrideLibrary();
        if (libName == null) {
            libName = findNativeLibrary();
            if (libName == null) {
                throw new IllegalStateException("Native library not found");
            }
        }

        Path nativeLibDir = Paths.get(libName).getParent();
        if (nativeLibDir == null || !nativeLibDir.toFile().isDirectory()) {
            throw new IllegalStateException("Native folder cannot be found");
        }
        String jniPath = copyJniLibraryFromClasspath(nativeLibDir);
        System.load(libName); // NOPMD
        logger.debug("Loading DLR native library from: {}", libName);
        System.load(jniPath); // NOPMD
        logger.debug("Loading DLR JNI library from: {}", jniPath);
    }

    private static synchronized String findNativeLibrary() {
        Platform platform = Platform.detectPlatform("dlr");
        if (platform.isPlaceholder()) {
            return downloadDlr(platform);
        }
        return copyNativeLibraryFromClasspath(platform);
    }

    private static String copyNativeLibraryFromClasspath(Platform platform) {
        Path tmp = null;
        try {
            String libName = System.mapLibraryName(NATIVE_LIB_NAME);
            Path cacheDir = getCacheDir(platform);
            Path path = cacheDir.resolve(libName);
            if (Files.exists(path)) {
                return path.toAbsolutePath().toString();
            }

            Path dlrCacheRoot = Utils.getEngineCacheDir("dlr");
            Files.createDirectories(dlrCacheRoot);
            tmp = Files.createTempDirectory(dlrCacheRoot, "tmp");
            for (String file : platform.getLibraries()) {
                String libPath = "native/lib/" + file;
                logger.info("Extracting {} to cache ...", libPath);
                try (InputStream is = ClassLoaderUtils.getResourceAsStream(libPath)) {
                    Files.copy(is, tmp.resolve(file), StandardCopyOption.REPLACE_EXISTING);
                }
            }

            Utils.moveQuietly(tmp, cacheDir);
            return path.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract DLR native library", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static String findLibraryInPath(String libPath) {
        String[] paths = libPath.split(File.pathSeparator);
        List<String> mappedLibNames;
        mappedLibNames = Collections.singletonList(System.mapLibraryName(NATIVE_LIB_NAME));

        for (String path : paths) {
            File p = new File(path);
            if (!p.exists()) {
                continue;
            }
            for (String name : mappedLibNames) {
                if (p.isFile() && p.getName().endsWith(name)) {
                    return p.getAbsolutePath();
                }

                File file = new File(path, name);
                if (file.exists() && file.isFile()) {
                    return file.getAbsolutePath();
                }
            }
        }
        return null;
    }

    private static String findNativeOverrideLibrary() {
        String libPath = Utils.getEnvOrSystemProperty("DLR_LIBRARY_PATH");
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

    private static String copyJniLibraryFromClasspath(Path nativeDir) {
        String name = System.mapLibraryName(LIB_NAME);
        Platform platform = Platform.detectPlatform("dlr");
        String classifier = platform.getClassifier();
        String djlVersion = platform.getApiVersion();
        Path path = nativeDir.resolve(djlVersion + '-' + name);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }
        Path tmp = null;
        // both cpu & gpu share the same jnilib
        String lib = "jnilib/" + classifier + '/' + name;
        try (InputStream is = ClassLoaderUtils.getResourceAsStream(lib)) {
            tmp = Files.createTempFile(nativeDir, "jni", "tmp");
            Files.copy(is, tmp, StandardCopyOption.REPLACE_EXISTING);
            Utils.moveQuietly(tmp, path);
            return path.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new IllegalStateException("Cannot copy jni files", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static String downloadDlr(Platform platform) {
        String version = platform.getVersion();
        String flavor = platform.getFlavor();
        String os = platform.getOsPrefix();

        String libName = System.mapLibraryName(NATIVE_LIB_NAME);
        Path cacheDir = getCacheDir(platform);
        Path path = cacheDir.resolve(libName);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }
        // if files not found
        Path dlrCacheRoot = Utils.getEngineCacheDir("dlr");

        Matcher matcher = VERSION_PATTERN.matcher(version);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Unexpected version: " + version);
        }
        String link = "https://publish.djl.ai/dlr-" + matcher.group(1) + "/native";
        Path tmp = null;
        try (InputStream is = Utils.openUrl(link + "/files.txt")) {
            Files.createDirectories(dlrCacheRoot);
            List<String> lines = Utils.readLines(is);
            if (flavor.startsWith("cu")
                    && !lines.contains(flavor + '/' + os + "/native/lib/" + libName)) {
                logger.warn("No matching cuda flavor for {} found: {}.", os, flavor);
                // fallback to CPU
                flavor = "cpu";

                // check again
                path = cacheDir.resolve(libName);
                if (Files.exists(path)) {
                    return path.toAbsolutePath().toString();
                }
            }

            tmp = Files.createTempDirectory(dlrCacheRoot, "tmp");
            boolean found = false;
            for (String line : lines) {
                if (line.startsWith(os + '/' + flavor + '/')) {
                    found = true;
                    URL url = new URL(link + '/' + line);
                    String fileName = line.substring(line.lastIndexOf('/') + 1);
                    logger.info("Downloading {} ...", url);
                    try (InputStream fis = Utils.openUrl(url)) {
                        Files.copy(fis, tmp.resolve(fileName), StandardCopyOption.REPLACE_EXISTING);
                    }
                }
            }
            if (!found) {
                throw new IllegalStateException(
                        "No DLR native library matches your operating system: " + platform);
            }

            Utils.moveQuietly(tmp, cacheDir);
            return path.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to download DLR native library", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static Path getCacheDir(Platform platform) {
        String version = platform.getVersion();
        String flavor = platform.getFlavor();
        String classifier = platform.getClassifier();
        Path cacheDir = Utils.getEngineCacheDir("dlr");
        logger.debug("Using cache dir: {}", cacheDir);
        return cacheDir.resolve(version + '-' + flavor + '-' + classifier);
    }
}
