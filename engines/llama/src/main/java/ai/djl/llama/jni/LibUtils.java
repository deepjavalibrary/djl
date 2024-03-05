/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.llama.jni;

import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Platform;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

/** Utilities for finding the llama.cpp native binary on the System. */
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = System.mapLibraryName("djl_llama");
    private static final String LLAMA_NAME = System.mapLibraryName("llama");

    private LibUtils() {}

    /** Loads llama.cpp native library. */
    public static void loadLibrary() {
        List<String> libs = new ArrayList<>(3);
        libs.add(LLAMA_NAME);
        libs.add(LIB_NAME);
        if (System.getProperty("os.name").startsWith("Mac")) {
            libs.add("ggml-metal.metal");
        }
        Path dir = copyJniLibraryFromClasspath(libs.toArray(new String[0]));
        logger.debug("Loading llama.cpp library from: {}", dir);

        for (int i = 0; i < 2; ++i) {
            String lib = libs.get(i);
            String path = dir.resolve(lib).toString();
            logger.debug("Loading native library: {}", path);
            String nativeHelper = System.getProperty("ai.djl.llama.native_helper");
            if (nativeHelper != null && !nativeHelper.isEmpty()) {
                ClassLoaderUtils.nativeLoad(nativeHelper, path);
            } else {
                System.load(path); // NOPMD
            }
        }
    }

    private static Path copyJniLibraryFromClasspath(String... libs) {
        Path cacheDir = Utils.getEngineCacheDir("llama");
        Platform platform = Platform.detectPlatform("llama");
        String classifier = platform.getClassifier();
        String version = platform.getVersion();
        Path dir = cacheDir.resolve(version + '-' + classifier);
        Path path = dir.resolve(LIB_NAME);
        logger.debug("Using cache dir: {}", dir);
        if (Files.exists(path)) {
            return dir.toAbsolutePath();
        }

        Path tmp = null;
        try {
            Files.createDirectories(cacheDir);
            tmp = Files.createTempDirectory(cacheDir, "tmp");

            for (String libName : libs) {
                String libPath = "native/lib/" + classifier + "/" + libName;
                logger.info("Extracting {} to cache ...", libPath);
                try (InputStream is = ClassLoaderUtils.getResourceAsStream(libPath)) {
                    Path target = tmp.resolve(libName);
                    Files.copy(is, target, StandardCopyOption.REPLACE_EXISTING);
                }
            }
            Utils.moveQuietly(tmp, dir);
            return dir.toAbsolutePath();
        } catch (IOException e) {
            throw new IllegalStateException("Cannot copy jni files", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }
}
