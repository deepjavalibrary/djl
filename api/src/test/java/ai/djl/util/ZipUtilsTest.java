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
package ai.djl.util;

import org.apache.commons.compress.archivers.zip.Zip64Mode;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class ZipUtilsTest {

    @Test
    public void testEmptyZipFile() throws IOException {
        Path path = Paths.get("build/empty.zip");
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(path))) {
            zos.finish();
        }

        Path output = Paths.get("build/output");
        Files.createDirectories(output);
        try (InputStream is = Files.newInputStream(path)) {
            ZipUtils.unzip(is, output);
        }
    }

    @Test
    public void testInvalidZipFile() throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        try (ZipOutputStream zos = new ZipOutputStream(bos)) {
            zos.putNextEntry(new ZipEntry("dir/"));
        }
        byte[] buf = bos.toByteArray();
        buf[32] = 'R';

        Path path = Paths.get("build/invalid.zip");
        try (OutputStream os = Files.newOutputStream(path)) {
            os.write(buf);
        }

        Path output = Paths.get("build/output");
        Files.createDirectories(output);
        try (InputStream is = Files.newInputStream(path)) {
            Assert.assertThrows(() -> ZipUtils.unzip(is, output));
        }

        buf[31] = '.';
        buf[32] = '.';
        path = Paths.get("build/relative_path.zip");
        try (OutputStream os = Files.newOutputStream(path)) {
            os.write(buf);
        }

        Files.createDirectories(output);
        try (InputStream is = Files.newInputStream(path)) {
            Assert.assertThrows(() -> ZipUtils.unzip(is, output));
        }
    }

    @Test(enabled = false)
    void testZip64() throws IOException {
        Path path = Paths.get("build/zip64.zip");
        try (OutputStream fos = Files.newOutputStream(path);
                ZipArchiveOutputStream zos = new ZipArchiveOutputStream(fos)) {
            zos.setUseZip64(Zip64Mode.Always);
            ZipArchiveEntry entry = new ZipArchiveEntry("test.txt");
            entry.setComment("comments");
            zos.putArchiveEntry(entry);
            zos.write("test".getBytes(StandardCharsets.UTF_8));
            zos.closeArchiveEntry();
        }

        Path output = Paths.get("build/output");
        Files.createDirectories(output);
        try (InputStream is = Files.newInputStream(path)) {
            ZipUtils.unzip(is, output);
        }
    }
}
