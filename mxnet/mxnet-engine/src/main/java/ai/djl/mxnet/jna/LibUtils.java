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
package ai.djl.mxnet.jna;

import ai.djl.util.Platform;
import ai.djl.util.Utils;
import com.sun.jna.Library;
import com.sun.jna.Native;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utilities for finding the MXNet Engine binary on the System.
 *
 * <p>The Engine will be searched for in a variety of locations in the following order:
 *
 * <ol>
 *   <li>In the path specified by the MXNET_LIBRARY_PATH environment variable
 *   <li>In a jar file location in the classpath. These jars can be created with the mxnet-native
 *       module.
 *   <li>In the python3 path. These can be installed using pip.
 *   <li>In the python path. These can be installed using pip.
 * </ol>
 */
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "mxnet";

    private static final Pattern VERSION_PATTERN =
            Pattern.compile("(\\d+\\.\\d+\\.\\d+(-[a-z]+)?)(-SNAPSHOT)?(-\\d+)?");

    private LibUtils() {}

    public static MxnetLibrary loadLibrary() {
        String libName = getLibName();
        logger.debug("Loading mxnet library from: {}", libName);

        if (System.getProperty("os.name").startsWith("Linux")) {
            Map<String, Integer> options = new ConcurrentHashMap<>();
            int rtld = 1; // Linux RTLD lazy + local
            options.put(Library.OPTION_OPEN_FLAGS, rtld);
            return Native.load(libName, MxnetLibrary.class, options);
        }

        return Native.load(libName, MxnetLibrary.class);
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
        String libPath = System.getenv("MXNET_LIBRARY_PATH");
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
        Enumeration<URL> urls;
        try {
            urls =
                    Thread.currentThread()
                            .getContextClassLoader()
                            .getResources("native/lib/mxnet.properties");
        } catch (IOException e) {
            logger.warn("", e);
            return null;
        }

        // No native jars
        if (!urls.hasMoreElements()) {
            logger.debug("mxnet.properties not found in class path.");
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
                    return downloadMxnet(placeholder);
                } catch (IOException e) {
                    throw new IllegalStateException("Failed to download MXNet native library", e);
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read MXNet native library jar properties", e);
        }

        throw new IllegalStateException(
                "Your MXNet native library jar does not match your operating system. Make sure that the Maven Dependency Classifier matches your system type.");
    }

    private static String loadLibraryFromClasspath(Platform platform) {
        Path tmp = null;
        try {
            String libName = System.mapLibraryName(LIB_NAME);
            Path cacheFolder = Utils.getEngineCacheDir("mxnet");
            String version = platform.getVersion();
            String flavor = platform.getFlavor();
            if (flavor.isEmpty()) {
                flavor = "mkl";
            } else if (!flavor.endsWith("mkl")) {
                flavor += "mkl"; // NOPMD
            }
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
                        throw new IllegalStateException("MXNet library not found: " + libPath);
                    }
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

    private static String findLibraryInPath(String libPath) {
        String[] paths = libPath.split(File.pathSeparator);
        List<String> mappedLibNames;
        if (com.sun.jna.Platform.isMac()) {
            mappedLibNames = Arrays.asList("libmxnet.dylib", "libmxnet.jnilib", "libmxnet.so");
        } else {
            mappedLibNames = Collections.singletonList(System.mapLibraryName(LIB_NAME));
        }

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

    private static String downloadMxnet(Platform platform) throws IOException {
        String version = platform.getVersion();
        String flavor = platform.getFlavor();
        if (flavor.isEmpty()) {
            flavor = "mkl";
        } else if (!flavor.endsWith("mkl")) {
            flavor += "mkl"; // NOPMD
        }
        String classifier = platform.getClassifier();
        String cudaArch = platform.getCudaArch();
        String os = platform.getOsPrefix();

        String libName = System.mapLibraryName(LIB_NAME);
        Path cacheFolder = Utils.getEngineCacheDir("mxnet");
        logger.debug("Using cache dir: {}", cacheFolder);
        Path dir = cacheFolder.resolve(version + '-' + flavor + '-' + classifier);
        Path path = dir.resolve(libName);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }

        Files.createDirectories(cacheFolder);

        Matcher matcher = VERSION_PATTERN.matcher(version);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Unexpected version: " + version);
        }

        Path tmp = Files.createTempDirectory(cacheFolder, "tmp");
        String link = "https://publish.djl.ai/mxnet-" + matcher.group(1);
        try (InputStream is = new URL(link + "/files.txt").openStream()) {
            List<String> lines = Utils.readLines(is);
            if (cudaArch != null) {
                // has CUDA
                if ("win".equals(os)) {
                    if (!lines.contains(os + '/' + flavor + "/mxnet_" + cudaArch + ".dll.gz")) {
                        logger.warn(
                                "No matching cuda flavor for {} found: {}/sm_{}.",
                                os,
                                flavor,
                                cudaArch);
                        // fallback to CPU
                        flavor = "mkl";
                    }
                } else if ("linux".equals(os)) {
                    if (!lines.contains(os + '/' + flavor + "/libmxnet.so.gz")
                            || notSupported(platform)) {
                        logger.warn(
                                "No matching cuda flavor for {} found: {}/sm_{}.",
                                os,
                                flavor,
                                cudaArch);
                        // fallback to CPU
                        flavor = "mkl";
                    }
                } else {
                    throw new AssertionError("Unsupported GPU operating system: " + os);
                }

                // check again in case fallback to cpu
                if ("mkl".equals(flavor)) {
                    dir = cacheFolder.resolve(version + '-' + flavor + '-' + classifier);
                    path = dir.resolve(libName);
                    if (Files.exists(path)) {
                        return path.toAbsolutePath().toString();
                    }
                }
            }

            for (String line : lines) {
                if (line.startsWith(os + "/common/") || line.startsWith(os + '/' + flavor + '/')) {
                    URL url = new URL(link + '/' + line);
                    String fileName = line.substring(line.lastIndexOf('/') + 1, line.length() - 3);
                    if ("win".equals(os)) {
                        if ("libmxnet.dll".equals(fileName)) {
                            fileName = "mxnet.dll";
                        } else if (fileName.startsWith("mxnet_")) {
                            if (!("mxnet_" + cudaArch + ".dll").equals(fileName)) {
                                continue;
                            }
                            fileName = "mxnet.dll"; // split CUDA build
                        }
                    }
                    logger.info("Downloading {} ...", fileName);
                    try (InputStream fis = new GZIPInputStream(url.openStream())) {
                        Files.copy(fis, tmp.resolve(fileName), StandardCopyOption.REPLACE_EXISTING);
                    }
                }
            }

            Utils.moveQuietly(tmp, dir);
            return path.toAbsolutePath().toString();
        } finally {
            Utils.deleteQuietly(tmp);
        }
    }

    private static boolean notSupported(Platform platform) {
        // mxnet-native-cu102mkl:1.8.0: 3.0, 5.0, 6.0, 7.0, 7.5
        // mxnet-native-cu110mkl:1.8.0: 5.0, 6.0, 7.0, 8.0
        if (platform.getVersion().startsWith("1.8.")) {
            String flavor = platform.getFlavor();
            String cudaArch = platform.getCudaArch();
            if ("cu110".equals(flavor)) {
                return !Arrays.asList("50", "60", "70", "80").contains(cudaArch);
            } else if ("cu102".equals(flavor)) {
                return !Arrays.asList("30", "50", "60", "70", "75").contains(cudaArch);
            }
        }
        return false;
    }
}
