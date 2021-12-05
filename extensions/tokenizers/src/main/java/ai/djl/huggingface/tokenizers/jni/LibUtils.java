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
package ai.djl.huggingface.tokenizers.jni;

import ai.djl.engine.EngineException;
import ai.djl.util.Platform;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Utilities for finding the Huggingface tokenizer native binary on the System. */
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = System.mapLibraryName("tokenizers");

    private static EngineException exception;

    static {
        try {
            loadLibrary();
        } catch (RuntimeException e) {
            exception = new EngineException("Failed to load Huggingface native library.", e);
        }
    }

    private LibUtils() {}

    public static void checkStatus() {
        if (exception != null) {
            throw exception;
        }
    }

    private static void loadLibrary() {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new UnsupportedOperationException("Windows is not supported.");
        }

        String libName = copyJniLibraryFromClasspath();
        logger.debug("Loading huggingface library from: {}", libName);
        System.load(libName); // NOPMD
    }

    private static String copyJniLibraryFromClasspath() {
        Path cacheDir = Utils.getEngineCacheDir("tokenizers");
        Platform platform = Platform.detectPlatform("tokenizers");
        String classifier = platform.getClassifier();
        String version = platform.getVersion();
        Path dir = cacheDir.resolve(version + '-' + classifier);
        Path path = dir.resolve(LIB_NAME);
        logger.debug("Using cache dir: {}", dir);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }
        Path tmp = null;
        String libPath = "/native/lib/" + classifier + "/" + LIB_NAME;
        logger.info("Extracting {} to cache ...", libPath);
        try (InputStream stream = LibUtils.class.getResourceAsStream(libPath)) {
            if (stream == null) {
                throw new IllegalStateException("Huggingface library not found: " + libPath);
            }
            Files.createDirectories(dir);
            tmp = Files.createTempFile(dir, "jni", "tmp");
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
