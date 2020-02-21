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

import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
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
@SuppressWarnings({
    "PMD.ClassWithOnlyPrivateConstructorsShouldBeFinal",
    "FinalClass",
    "MissingJavadocMethod"
})
public class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "mxnet";
    private static final Pattern PATH_PATTERN = Pattern.compile("\\s*'(.+)',");

    private LibUtils() {}

    public static MxnetLibrary loadLibrary() {
        String libName = getLibName();
        logger.debug("Loading mxnet library from: {}", libName);

        return Native.load(libName, MxnetLibrary.class);
    }

    public static String getLibName() {
        String libName = LibUtils.findOverrideLibrary();
        if (libName == null) {
            libName = LibUtils.findLibraryInClasspath();
            if (libName == null) {
                libName = searchPythonPath("python3 -m site");
                if (libName == null) {
                    libName = searchPythonPath("python -m site");
                    if (libName == null) {
                        libName = LIB_NAME;
                    }
                }
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
        URL url = LibUtils.class.getResource("/native/lib/mxnet.properties");
        if (url == null) {
            return null;
        }

        Path tmp = null;
        try (InputStream conf = url.openStream()) {
            Properties prop = new Properties();
            prop.load(conf);
            // 1.6.0-c later should always has version property
            String version = prop.getProperty("version", "1.6.0-c");
            String osName = System.getProperty("os.name");
            String osPrefix;
            if (osName.startsWith("Win")) {
                osPrefix = "win";
            } else if (osName.startsWith("Mac")) {
                osPrefix = "osx";
            } else if (osName.startsWith("Linux")) {
                osPrefix = "linux";
            } else {
                throw new AssertionError("Unsupported platform: " + osName);
            }

            if (prop.getProperty("placeholder") != null) {
                try {
                    return downloadMxnet(version, osPrefix);
                } catch (IOException e) {
                    throw new IllegalStateException("Failed to download MXNet native library", e);
                }
            }
            String classifier = prop.getProperty("classifier", "");
            if (!classifier.isEmpty()) {
                if (!osPrefix.equals(classifier.split("-")[1])) {
                    throw new IllegalStateException(
                            "Your MXNet native library jar does not match your operating system. Make sure that the Maven Dependency Classifier matches your system type.");
                }
            }
            String libs = prop.getProperty("libraries");
            String[] files = libs.split(",");

            String userHome = System.getProperty("user.home");
            String libName = System.mapLibraryName(LIB_NAME);
            Path dir = Paths.get(userHome, ".mxnet/cache/" + version + classifier);
            Path path = dir.resolve(libName);
            if (Files.exists(path)) {
                return path.toAbsolutePath().toString();
            }
            tmp = Paths.get(userHome, ".mxnet/cache/tmp");
            Files.createDirectories(tmp);
            for (String file : files) {
                String libPath = "/native/lib/" + file;
                try (InputStream is = LibUtils.class.getResourceAsStream(libPath)) {
                    Files.copy(is, tmp.resolve(file));
                }
            }

            Utils.deleteQuietly(dir);
            Files.move(tmp, dir);
            tmp = null;
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
        if (Platform.isMac()) {
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

    private static String searchPythonPath(String cmd) {
        String libName;
        if (Platform.isMac()) {
            // pip package use libmxnet.so instead of libmxnet.dylib, JNA by default only
            // load .dylib file, we have to use absolute path to load libmxnet.so
            libName = "libmxnet.so";
        } else {
            libName = System.mapLibraryName(LIB_NAME);
        }

        try {
            Process process = Runtime.getRuntime().exec(cmd);
            try (BufferedReader reader =
                    new BufferedReader(
                            new InputStreamReader(
                                    process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    Matcher m = PATH_PATTERN.matcher(line);
                    if (m.matches()) {
                        File dir = new File(m.group(1));
                        if (dir.isDirectory()) {
                            File file = new File(dir, "mxnet/" + libName);
                            if (file.exists()) {
                                return file.getAbsolutePath();
                            }
                        }
                    }
                }
            }
        } catch (IOException e) {
            if (logger.isTraceEnabled()) {
                logger.trace("Failed execute cmd: " + cmd, e);
            }
        }
        return null;
    }

    private static String downloadMxnet(String version, String os) throws IOException {
        String flavor;
        String cudaArch;
        if (CudaUtils.getGpuCount() > 0) {
            flavor = "cu" + CudaUtils.getCudaVersionString() + "mkl";
            cudaArch = CudaUtils.getComputeCapability(0);
        } else {
            flavor = "mkl";
            cudaArch = null;
        }

        String classifier = os + "-x86_64";
        String userHome = System.getProperty("user.home");
        String libName = System.mapLibraryName(LIB_NAME);
        Path dir = Paths.get(userHome, ".mxnet/cache/" + version + flavor + '-' + classifier);
        Path path = dir.resolve(libName);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }

        Path tmp = Paths.get(userHome, ".mxnet/cache/tmp");
        Files.createDirectories(tmp);

        int pos = version.indexOf("-SNAPSHOT");
        if (pos > 0) {
            version = version.substring(0, pos);
        }
        String link = "https://djl-ai.s3.amazonaws.com/publish/mxnet-" + version;
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
                    if (!lines.contains(os + '/' + flavor + "/libmxnet.so.gz")) {
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
            }

            for (String line : lines) {
                if (line.startsWith(os + "/common/") || line.startsWith(os + '/' + flavor + '/')) {
                    URL url = new URL(link + '/' + line);
                    String fileName = line.substring(line.lastIndexOf('/') + 1, line.length() - 3);
                    if ("win".equals(os)) {
                        if ("libmxnet.dll".equals(fileName)) {
                            continue;
                        } else if ("libcumxnet.dll".equals(fileName)) {
                            fileName = "mxnet.dll";
                        } else if (fileName.startsWith("mxnet_")
                                && !("mxnet_" + cudaArch + ".dll").equals(fileName)) {
                            continue;
                        }
                    }
                    try (InputStream fis = new GZIPInputStream(url.openStream())) {
                        Files.copy(fis, tmp.resolve(fileName));
                    }
                }
            }

            Utils.deleteQuietly(dir);
            Files.move(tmp, dir);
            tmp = null;
            return path.toAbsolutePath().toString();
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }
}
