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
package ai.djl.repository;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/** Utilities for working with zip files. */
public final class ZipUtils {

    private ZipUtils() {}

    /**
     * Unzips an input stream to a given path.
     *
     * @param is the input stream to unzip
     * @param dest the path to store the unzipped files
     * @throws IOException for failures to unzip the input stream and create files in the dest path
     */
    public static void unzip(InputStream is, Path dest) throws IOException {
        ZipInputStream zis = new ZipInputStream(is);
        ZipEntry entry;
        while ((entry = zis.getNextEntry()) != null) {
            String name = entry.getName();
            Path file = dest.resolve(name).toAbsolutePath();
            if (entry.isDirectory()) {
                Files.createDirectories(file);
            } else {
                Path parentFile = file.getParent();
                if (parentFile == null) {
                    throw new AssertionError(
                            "Parent path should never be null: " + file.toString());
                }
                Files.createDirectories(parentFile);
                Files.copy(zis, file);
            }
        }
    }
}
