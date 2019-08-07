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
package software.amazon.ai.integration.util;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class FileUtils {

    private FileUtils() {}

    public static void download(String source, String destination, String fileName)
            throws IOException {
        URL url = new URL(source);
        InputStream in = url.openStream();
        File destDir = new File(destination);
        if (!destDir.exists()) {
            if (!destDir.mkdir()) {
                throw new IOException("Failed to create directory: " + destDir);
            }
        }
        Files.copy(
                in, Paths.get(destination + "/" + fileName), StandardCopyOption.REPLACE_EXISTING);
    }

    public static void unzip(String zipFilePath, String dest) throws IOException {
        try (ZipInputStream zis =
                new ZipInputStream(Files.newInputStream(Paths.get(zipFilePath)))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                String name = entry.getName();
                Path file = Paths.get(dest).resolve(name);
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

    public static void deleteFileOrDir(String target) {
        try {
            Files.walk(Paths.get(target))
                    .sorted(Comparator.reverseOrder())
                    .forEach(
                            path -> {
                                try {
                                    Files.deleteIfExists(path);
                                } catch (IOException ignore) {
                                    // ignore
                                }
                            });
        } catch (IOException ignore) {
            // ignore
        }
    }
}
