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
package ai.djl.pytorch.jni;

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
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utilities for finding the PyTorch Engine binary on the System.
 *
 * <p>The Engine will be searched for in a variety of locations in the following order:
 *
 * <ol>
 *   <li>In the path specified by the PYTORCH_LIBRARY_PATH environment variable
 *   <li>In a jar file location in the classpath. These jars can be created with the pytorch-native
 *       module.
 * </ol>
 */
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "djl_torch";
    private static final String NATIVE_LIB_NAME = "torch";

    private static final Pattern VERSION_PATTERN =
            Pattern.compile("(\\d+\\.\\d+\\.\\d+(-\\w)?)(-SNAPSHOT)?(-\\d+)?");

    private LibUtils() {}

    public static void loadLibrary() {
        // TODO workaround to make it work on Android Studio
        // It should search for several places to find the native library
        if (System.getProperty("java.vendor.url").equals("http://www.android.com/")) {
            System.loadLibrary(LIB_NAME); // NOPMD
            return;
        }
        String libName = findOverrideLibrary();
        if (libName == null) {
            String nativeLibDir = findNativeLibrary();
            if (nativeLibDir != null) {
                libName = copyJniLibraryFromClasspath(Paths.get(nativeLibDir));
            } else {
                throw new IllegalStateException("Native library not found");
            }
        }
        logger.debug("Loading pytorch library from: {}", libName);
        if (System.getProperty("os.name").startsWith("Win")) {
            loadWinDependencies(libName);
        }
        System.load(libName); // NOPMD
    }

    private static void loadWinDependencies(String libName) {
        Path libDir = Paths.get(libName).getParent();
        if (libDir == null) {
            throw new IllegalArgumentException("Invalid library path!");
        }
        try (Stream<Path> paths = Files.walk(libDir)) {
            paths.filter(
                            path -> {
                                String name = path.getFileName().toString();
                                return !"c10_cuda.dll".equals(name)
                                        && !"torch.dll".equals(name)
                                        && !"torch_cpu.dll".equals(name)
                                        && !"torch_cuda.dll".equals(name)
                                        && !"fbgemm.dll".equals(name)
                                        && Files.isRegularFile(path)
                                        && !name.endsWith("djl_torch.dll");
                            })
                    .map(path -> path.toAbsolutePath().toString())
                    .forEach(System::load);
            System.load(libDir.resolve("fbgemm.dll").toAbsolutePath().toString());
            System.load(libDir.resolve("torch_cpu.dll").toAbsolutePath().toString());
            if (Files.exists(libDir.resolve("c10_cuda.dll"))) {
                // Windows System.load is global load
                System.load(libDir.resolve("c10_cuda.dll").toAbsolutePath().toString());
                System.load(libDir.resolve("torch_cuda.dll").toAbsolutePath().toString());
            }
            System.load(libDir.resolve("torch.dll").toAbsolutePath().toString());
        } catch (IOException e) {
            throw new IllegalArgumentException("Folder not exist! " + libDir, e);
        }
    }

    private static String findOverrideLibrary() {
        String libPath = System.getenv("PYTORCH_LIBRARY_PATH");
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

    private static String findLibraryInPath(String libPath) {
        String[] paths = libPath.split(File.pathSeparator);
        List<String> mappedLibNames;
        mappedLibNames = Collections.singletonList(System.mapLibraryName(LIB_NAME));

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

    private static String copyJniLibraryFromClasspath(Path nativeDir) {
        String name = System.mapLibraryName(LIB_NAME);
        Platform platform = Platform.fromSystem();
        String classifier = platform.getClassifier();
        String flavor = platform.getFlavor();
        if (flavor.isEmpty()) {
            flavor = "cpu";
        }
        Properties prop = new Properties();
        try (InputStream stream =
                LibUtils.class.getResourceAsStream(
                        "/jnilib/" + classifier + "/" + flavor + "/pytorch.properties")) {
            prop.load(stream);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot find pytorch property file", e);
        }
        String version = prop.getProperty("version");
        Path path = nativeDir.resolve(version + flavor + name);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }

        Path tmp = null;
        try (InputStream stream =
                LibUtils.class.getResourceAsStream(
                        "/jnilib/" + classifier + "/" + flavor + "/" + name)) {
            tmp = Files.createTempFile(nativeDir, "jni", "tmp");
            Files.copy(stream, tmp, StandardCopyOption.REPLACE_EXISTING);
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

    private static synchronized String findNativeLibrary() {
        Enumeration<URL> urls;
        try {
            urls =
                    Thread.currentThread()
                            .getContextClassLoader()
                            .getResources("native/lib/pytorch.properties");
        } catch (IOException e) {
            logger.warn("", e);
            return null;
        }
        // No native jars
        if (!urls.hasMoreElements()) {
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
                    return downloadPyTorch(placeholder);
                } catch (IOException e) {
                    throw new IllegalStateException("Failed to download PyTorch native library", e);
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read PyTorch native library jar properties", e);
        }
        throw new IllegalStateException(
                "Your PyTorch native library jar does not match your operating system. Make sure the Maven Dependency Classifier matches your system type.");
    }

    private static String copyNativeLibraryFromClasspath(Platform platform) {
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
                return dir.toAbsolutePath().toString();
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
            return dir.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract PyTorch native library", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static String downloadPyTorch(Platform platform) throws IOException {
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
            return dir.toAbsolutePath().toString();
        }
        // if files not found
        Files.createDirectories(cacheDir);
        Path tmp = Files.createTempDirectory(cacheDir, "tmp");

        Matcher matcher = VERSION_PATTERN.matcher(version);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Unexpected version: " + version);
        }
        String link = "https://djl-ai.s3.amazonaws.com/publish/pytorch-" + matcher.group(1);
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
            return dir.toAbsolutePath().toString();
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
                        return Paths.get(userHome, ".pytorch/cache");
                    }
                }
                return Paths.get(cacheDir, "pytorch");
            }
        }
        return Paths.get(cacheDir, ".pytorch/cache");
    }
}
