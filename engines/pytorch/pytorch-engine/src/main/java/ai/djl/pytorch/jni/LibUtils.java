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
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
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
            Pattern.compile("(\\d+\\.\\d+\\.\\d+(-[a-z]+)?)(-SNAPSHOT)?(-\\d+)?");

    private static String version;
    private static String libtorchPath;

    private LibUtils() {}

    public static void loadLibrary() {
        Properties prop = new Properties();
        try (InputStream stream =
                LibUtils.class.getResourceAsStream("/jnilib/pytorch.properties")) {
            prop.load(stream);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot find pytorch property file", e);
        }
        String djlVersion = prop.getProperty("version");
        String jniVersion = prop.getProperty("jni_version");

        version = jniVersion; // default to bundled jni version

        // TODO workaround to make it work on Android Studio
        // It should search for several places to find the native library
        if ("http://www.android.com/".equals(System.getProperty("java.vendor.url"))) {
            System.loadLibrary(LIB_NAME); // NOPMD
            return;
        }
        libtorchPath = getLibName(djlVersion, jniVersion);
        logger.debug("Loading pytorch library from: {}", libtorchPath);
        if (System.getProperty("os.name").startsWith("Win")) {
            loadWinDependencies(libtorchPath);
        }
        loadNativeLibrary(libtorchPath);
    }

    public static String getLibName(String djlVersion, String jniVersion) {
        String libName = findOverrideLibrary();
        if (libName == null) {
            AtomicBoolean fallback = new AtomicBoolean(false);
            String nativeLibDir = findNativeLibrary(fallback);
            if (nativeLibDir != null) {
                Path nativeDir = Paths.get(nativeLibDir);
                libName =
                        copyJniLibraryFromClasspath(
                                nativeDir, djlVersion, jniVersion, fallback.get());
            } else {
                throw new IllegalStateException("Native library not found");
            }
        }
        return libName;
    }

    public static String getVersion() {
        return version;
    }

    public static String getLibtorchPath() {
        return libtorchPath;
    }

    private static void loadWinDependencies(String libName) {
        Path libDir = Paths.get(libName).getParent();
        if (libDir == null) {
            throw new IllegalArgumentException("Invalid library path!");
        }

        Set<String> loadLater =
                new HashSet<>(
                        Arrays.asList(
                                "c10_cuda.dll",
                                "torch.dll",
                                "torch_cpu.dll",
                                "torch_cuda.dll",
                                "torch_cuda_cpp.dll",
                                "torch_cuda_cu.dll",
                                "fbgemm.dll"));

        try (Stream<Path> paths = Files.walk(libDir)) {
            paths.filter(
                            path -> {
                                String name = path.getFileName().toString();
                                return !loadLater.contains(name)
                                        && Files.isRegularFile(path)
                                        && !name.endsWith("djl_torch.dll")
                                        && !name.startsWith("cudnn");
                            })
                    .map(path -> path.toAbsolutePath().toString())
                    .forEach(System::load);
            loadNativeLibrary(libDir.resolve("fbgemm.dll").toAbsolutePath().toString());
            loadNativeLibrary(libDir.resolve("torch_cpu.dll").toAbsolutePath().toString());
            if (Files.exists(libDir.resolve("c10_cuda.dll"))) {
                if (Files.exists((libDir.resolve("cudnn64_8.dll")))) {
                    loadNativeLibrary(libDir.resolve("cudnn64_8.dll").toAbsolutePath().toString());
                    loadNativeLibrary(
                            libDir.resolve("cudnn_ops_infer64_8.dll").toAbsolutePath().toString());
                    loadNativeLibrary(
                            libDir.resolve("cudnn_ops_train64_8.dll").toAbsolutePath().toString());
                    loadNativeLibrary(
                            libDir.resolve("cudnn_cnn_infer64_8.dll").toAbsolutePath().toString());
                    loadNativeLibrary(
                            libDir.resolve("cudnn_cnn_train64_8.dll").toAbsolutePath().toString());
                    loadNativeLibrary(
                            libDir.resolve("cudnn_adv_infer64_8.dll").toAbsolutePath().toString());
                    loadNativeLibrary(
                            libDir.resolve("cudnn_adv_train64_8.dll").toAbsolutePath().toString());
                } else if (Files.exists((libDir.resolve("cudnn64_7.dll")))) {
                    loadNativeLibrary(libDir.resolve("cudnn64_7.dll").toAbsolutePath().toString());
                }
                // Windows System.load is global load
                loadNativeLibrary(libDir.resolve("c10_cuda.dll").toAbsolutePath().toString());
                // workaround for CU111, these files not exist in CU102
                if (Files.exists(libDir.resolve("torch_cuda_cpp.dll"))) {
                    loadNativeLibrary(
                            libDir.resolve("torch_cuda_cpp.dll").toAbsolutePath().toString());
                    loadNativeLibrary(
                            libDir.resolve("torch_cuda_cu.dll").toAbsolutePath().toString());
                }
                loadNativeLibrary(libDir.resolve("torch_cuda.dll").toAbsolutePath().toString());
            }
            loadNativeLibrary(libDir.resolve("torch.dll").toAbsolutePath().toString());
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

    private static String copyJniLibraryFromClasspath(
            Path nativeDir, String djlVersion, String jniVersion, boolean fallback) {
        String name = System.mapLibraryName(LIB_NAME);
        Platform platform = Platform.fromSystem();
        String classifier = platform.getClassifier();
        String flavor = platform.getFlavor();
        if (fallback) {
            flavor = "cpu";
        }
        Path precxx11Lib = nativeDir.resolve("libstdc++.so.6");
        if (Files.exists(precxx11Lib)) {
            flavor += "-precxx11"; // NOPMD
            logger.info("Using precxx11 jnilib.");
        }

        Path path = nativeDir.resolve(djlVersion + '-' + flavor + '-' + name);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }

        if (!version.startsWith(jniVersion)) {
            downloadJniLib(nativeDir, path, djlVersion, classifier, flavor, name);
            return path.toAbsolutePath().toString();
        }

        Path tmp = null;
        String libPath = "/jnilib/" + classifier + '/' + flavor + '/' + name;
        logger.info("Extracting {} to cache ...", libPath);
        try (InputStream stream = LibUtils.class.getResourceAsStream(libPath)) {
            if (stream == null) {
                throw new IllegalStateException("PyTorch jni not found: " + libPath);
            }
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

    private static synchronized String findNativeLibrary(AtomicBoolean fallback) {
        String overrideVersion = System.getenv("PYTORCH_VERSION");
        if (overrideVersion == null) {
            overrideVersion = System.getProperty("PYTORCH_VERSION");
        }
        if (overrideVersion != null) {
            version = overrideVersion;
            Platform auto = Platform.fromSystem(overrideVersion);
            return downloadPyTorch(auto, fallback);
        }
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
            Platform auto = Platform.fromSystem(version);
            return downloadPyTorch(auto, fallback);
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
                } else if (platform.matches(systemPlatform, false)) {
                    matching = platform;
                    break;
                }
            }

            if (matching != null) {
                if ("cpu".equals(matching.getFlavor())) {
                    fallback.set(true);
                }
                return copyNativeLibraryFromClasspath(matching);
            }

            if (placeholder != null) {
                return downloadPyTorch(placeholder, fallback);
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
        version = platform.getVersion();
        String flavor = platform.getFlavor();
        // TODO: include precxx11 into native jar's flavor property
        if (Arrays.asList(platform.getLibraries()).contains("libstdc++.so.6")) {
            flavor += "-precxx11"; // NOPMD
        }
        String classifier = platform.getClassifier();
        try {
            String libName = System.mapLibraryName(NATIVE_LIB_NAME);
            Path cacheDir = Utils.getEngineCacheDir("pytorch");
            logger.debug("Using cache dir: {}", cacheDir);
            Path dir = cacheDir.resolve(version + '-' + flavor + '-' + classifier);
            Path path = dir.resolve(libName);
            if (Files.exists(path)) {
                return dir.toAbsolutePath().toString();
            }

            Files.createDirectories(cacheDir);
            tmp = Files.createTempDirectory(cacheDir, "tmp");
            for (String file : platform.getLibraries()) {
                String libPath = "/native/lib/" + file;
                logger.info("Extracting {} to cache ...", libPath);
                try (InputStream is = LibUtils.class.getResourceAsStream(libPath)) {
                    if (is == null) {
                        throw new IllegalStateException("PyTorch library not found: " + libPath);
                    }
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

    private static void loadNativeLibrary(String path) {
        String nativeHelper = System.getProperty("ai.djl.pytorch.native_helper");
        if (nativeHelper != null && !nativeHelper.isEmpty()) {
            try {
                Class<?> clazz = Class.forName(nativeHelper);
                Method method = clazz.getDeclaredMethod("load", String.class);
                method.invoke(null, path);
            } catch (ReflectiveOperationException e) {
                throw new IllegalArgumentException("Invalid native_helper: " + nativeHelper, e);
            }
        }
        System.load(path); // NOPMD
    }

    private static String downloadPyTorch(Platform platform, AtomicBoolean fallback) {
        version = platform.getVersion();
        String flavor = platform.getFlavor();
        String classifier = platform.getClassifier();
        String os = platform.getOsPrefix();
        if (Boolean.getBoolean("PYTORCH_PRECXX11")
                || Boolean.parseBoolean(System.getenv("PYTORCH_PRECXX11"))) {
            flavor += "-precxx11";
        }

        String libName = System.mapLibraryName(NATIVE_LIB_NAME);
        Path cacheDir = Utils.getEngineCacheDir("pytorch");
        Path dir = cacheDir.resolve(version + '-' + flavor + '-' + classifier);
        logger.debug("Using cache dir: {}", dir);
        Path path = dir.resolve(libName);
        if (Files.exists(path)) {
            return dir.toAbsolutePath().toString();
        }

        Matcher matcher = VERSION_PATTERN.matcher(version);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Unexpected version: " + version);
        }
        String link = "https://publish.djl.ai/pytorch-" + matcher.group(1);
        Path tmp = null;
        try (InputStream is = new URL(link + "/files.txt").openStream()) {
            // if files not found
            Files.createDirectories(cacheDir);

            List<String> lines = Utils.readLines(is);
            if (!lines.contains(flavor + '/' + os + "/native/lib/" + libName + ".gz")) {
                if (flavor.startsWith("cu")) {
                    logger.warn("No matching cuda flavor for {} found: {}.", os, flavor);
                    // fallback to CPU
                    flavor = "cpu";
                    fallback.set(true);

                    // check again
                    dir = cacheDir.resolve(version + '-' + flavor + '-' + classifier);
                    path = dir.resolve(libName);
                    if (Files.exists(path)) {
                        return dir.toAbsolutePath().toString();
                    }
                } else {
                    throw new IOException("No matching flavor for " + os + " found: " + flavor);
                }
            }

            tmp = Files.createTempDirectory(cacheDir, "tmp");
            for (String line : lines) {
                if (line.startsWith(flavor + '/' + os + '/')) {
                    URL url = new URL(link + '/' + line);
                    String fileName = line.substring(line.lastIndexOf('/') + 1, line.length() - 3);
                    fileName = URLDecoder.decode(fileName, "UTF-8");
                    logger.info("Downloading {} ...", url);
                    try (InputStream fis = new GZIPInputStream(url.openStream())) {
                        Files.copy(fis, tmp.resolve(fileName), StandardCopyOption.REPLACE_EXISTING);
                    }
                }
            }

            Utils.moveQuietly(tmp, dir);
            return dir.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to download PyTorch native library", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static void downloadJniLib(
            Path nativeDir,
            Path path,
            String djlVersion,
            String classifier,
            String flavor,
            String name) {
        Matcher matcher = VERSION_PATTERN.matcher(djlVersion);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Unexpected djl version: " + djlVersion);
        }
        djlVersion = matcher.group(1);
        matcher = VERSION_PATTERN.matcher(version);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Unexpected pytorch version: " + version);
        }

        StringBuilder sb = new StringBuilder(100);
        sb.append("https://publish.djl.ai/pytorch-").append(matcher.group(1)).append("/jnilib/");
        if (flavor.contains("-precxx11")) {
            flavor = flavor.substring(0, flavor.length() - 9);
            sb.append("precxx11/");
        }
        sb.append(djlVersion)
                .append('/')
                .append(classifier)
                .append('/')
                .append(flavor)
                .append('/')
                .append(name);
        logger.info("Downloading jni {} to cache ...", sb);
        Path tmp = null;
        try (InputStream is = new URL(sb.toString()).openStream()) {
            tmp = Files.createTempFile(nativeDir, "jni", "tmp");
            Files.copy(is, tmp, StandardCopyOption.REPLACE_EXISTING);
            Utils.moveQuietly(tmp, path);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot download jni files", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }
}
