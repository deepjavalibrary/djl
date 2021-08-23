/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.tensorrt.jni;

import ai.djl.util.Platform;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Properties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utilities for finding the TensorRT Engine binary on the System.
 *
 * <p>The Engine will be searched for in a variety of locations in the following order:
 *
 * <ol>
 *   <li>In the path specified by the TENSORRT_LIBRARY_PATH environment variable
 * </ol>
 */
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "djl_trt";

    private LibUtils() {}

    public static void loadLibrary() {
        if (!System.getProperty("os.name").startsWith("Linux")) {
            throw new UnsupportedOperationException("TensorRT only supports Linux.");
        }
        String libName = copyJniLibraryFromClasspath();
        logger.debug("Loading TensorRT JNI library from: {}", libName);
        System.load(libName); // NOPMD
    }

    private static String copyJniLibraryFromClasspath() {
        String name = System.mapLibraryName(LIB_NAME);
        Platform platform = Platform.fromSystem();
        String classifier = platform.getClassifier();
        Properties prop = new Properties();
        try (InputStream stream =
                LibUtils.class.getResourceAsStream("/jnilib/tensorrt.properties")) {
            prop.load(stream);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot find TensorRT property file", e);
        }
        String version = prop.getProperty("version");
        Path cacheDir = Utils.getEngineCacheDir("tensorrt");
        Path dir = cacheDir.resolve(version + '-' + classifier);
        Path path = dir.resolve(name);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }
        Path tmp = null;
        String libPath = "/jnilib/" + classifier + "/" + name;
        logger.info("Extracting {} to cache ...", libPath);
        try (InputStream stream = LibUtils.class.getResourceAsStream(libPath)) {
            if (stream == null) {
                throw new IllegalStateException("TensorRT library not found: " + libPath);
            }
            Files.createDirectories(dir);
            tmp = Files.createTempFile(cacheDir, "jni", "tmp");
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
}
