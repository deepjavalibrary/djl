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

package ai.djl.tensorflow.engine;

import ai.djl.engine.EngineException;
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
import java.util.Arrays;
import java.util.List;
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
            Pattern.compile("(\\d+\\.\\d+\\.\\d+(-\\w)?)(-SNAPSHOT)?(-\\d+)?");
    private static final String[] SUPPORTED_CUDA_VERSIONS = {"cu102", "cu101", "cu100", "cu92"};

    private LibUtils() {}

    public static void loadLibrary() {
        String libName = getTensorFlowLib();
        if (libName != null) {
            logger.debug("Loading TensorFlow library from: {}", libName);
            String path = new File(libName).getParentFile().toString();
            System.setProperty("org.bytedeco.javacpp.platform.preloadpath", path);
        }
    }

    private static String getTensorFlowLib() {
        URL url =
                Thread.currentThread()
                        .getContextClassLoader()
                        .getResource("native/lib/tensorflow.properties");
        if (url == null) {
            // defer to tensorflow-core-api to handle loading native library.
            return null;
        }

        try {
            Platform platform = Platform.fromUrl(url);
            return downloadTensorFlow(platform);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to download TensorFlow native library", e);
        }
    }

    private static String downloadTensorFlow(Platform platform) throws IOException {
        String version = platform.getVersion();
        String os = platform.getOsPrefix();
        String classifier = platform.getClassifier();
        String cudaArch = platform.getCudaArch();
        String flavor = platform.getFlavor();
        if (flavor.isEmpty()) {
            flavor = "cpu";
        } else if (Arrays.asList(SUPPORTED_CUDA_VERSIONS).contains(flavor)) {
            // we don't need flavor specific binaries for TensorFlow
            flavor = "gpu";
        } else {
            logger.warn("Unsupported GPU platform: {}, fallback to CPU.", flavor);
            flavor = "cpu";
        }

        Path userHome = Paths.get(System.getProperty("user.home"));
        String libName = System.mapLibraryName(LIB_NAME);
        Path dir =
                userHome.resolve(".tensorflow/cache/" + version + '-' + flavor + '-' + classifier);
        Path path = dir.resolve(libName);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }

        Path tmp = userHome.resolve(".tensorflow/cache/tmp");
        Files.createDirectories(tmp);

        Matcher matcher = VERSION_PATTERN.matcher(version);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Unexpected version: " + version);
        }
        String link = "https://djl-ai.s3.amazonaws.com/publish/tensorflow-" + matcher.group(1);
        try (InputStream is = new URL(link + "/files.txt").openStream()) {
            List<String> lines = Utils.readLines(is);
            boolean found = downloadFiles(lines, link, os, flavor, tmp);
            if (!found && cudaArch != null) {
                // fallback to cpu
                flavor = "cpu";
                dir =
                        userHome.resolve(
                                ".tensorflow/cache/" + version + '-' + flavor + '-' + classifier);
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
                throw new EngineException(
                        "TensorFlow engine does not support this platform: " + os);
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

    private static boolean downloadFiles(
            List<String> lines, String link, String os, String flavor, Path tmp)
            throws IOException {
        boolean found = false;
        for (String line : lines) {
            if (line.startsWith(os + '/' + flavor + '/')) {
                found = true;
                URL url = new URL(link + '/' + line.replace("+", "%2B"));
                String fileName = line.substring(line.lastIndexOf('/') + 1, line.length() - 3);
                try (InputStream fis = new GZIPInputStream(url.openStream())) {
                    Files.copy(fis, tmp.resolve(fileName), StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
        return found;
    }
}
