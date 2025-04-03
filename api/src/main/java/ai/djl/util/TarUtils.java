/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.util;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.input.CloseShieldInputStream;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Utilities for working with zip files. */
public final class TarUtils {

    private TarUtils() {}

    /**
     * Un-compress a tar ball from InputStream.
     *
     * @param is the InputStream
     * @param dir the target directory
     * @param gzip if the bar ball is gzip
     * @throws IOException for failures to untar the input directory
     */
    public static void untar(InputStream is, Path dir, boolean gzip) throws IOException {
        InputStream bis;
        if (gzip) {
            bis = new GzipCompressorInputStream(new BufferedInputStream(is));
        } else {
            bis = new BufferedInputStream(is);
        }
        bis = CloseShieldInputStream.wrap(bis);
        try (TarArchiveInputStream tis = new TarArchiveInputStream(bis)) {
            TarArchiveEntry entry;
            while ((entry = tis.getNextEntry()) != null) {
                String entryName = entry.getName();
                ZipUtils.validateArchiveEntry(entryName, dir);
                Path file = dir.resolve(entryName).toAbsolutePath();
                if (entry.isDirectory()) {
                    Files.createDirectories(file);
                } else {
                    Path parentFile = file.getParent();
                    if (parentFile == null) {
                        throw new AssertionError("Parent path should never be null: " + file);
                    }
                    Files.createDirectories(parentFile);
                    Files.copy(tis, file, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
    }
}
