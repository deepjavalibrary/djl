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
package ai.djl.util;

import ai.djl.util.cuda.CudaUtils;

import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;

public class PlatformTest {

    @AfterClass
    public void tierDown() {
        Utils.deleteQuietly(Paths.get("build/tmp/testFile/"));
    }

    @Test
    public void testPlatform() throws IOException {
        URL invalid = createPropertyFile("");
        Assert.assertThrows(IllegalArgumentException.class, () -> Platform.fromUrl(invalid));

        URL url = createPropertyFile("version=1.8.0\nclassifier=cu113-linux-x86_64");
        // Use cu113 as target machine
        Platform system = Platform.fromUrl(url);
        Assert.assertEquals(system.getFlavor(), "cu113");
        Assert.assertEquals(system.getClassifier(), "linux-x86_64");
        Assert.assertEquals(system.getOsPrefix(), "linux");
        Assert.assertEquals(system.getOsArch(), "x86_64");

        url = createPropertyFile("version=1.8.0\nflavor=cpu-precxx11\nclassifier=linux-x86_64");
        Platform platform = Platform.fromUrl(url);
        Assert.assertEquals(platform.getFlavor(), "cpu-precxx11");
        Assert.assertEquals(platform.getClassifier(), "linux-x86_64");
        Assert.assertEquals(platform.getOsPrefix(), "linux");
        Assert.assertEquals(platform.getOsArch(), "x86_64");
        // cpu should always match with system
        Assert.assertTrue(platform.matches(system));
        Assert.assertFalse(system.matches(platform));

        url = createPropertyFile("version=1.8.0\nplaceholder=true");
        platform = Platform.fromUrl(url);
        Assert.assertTrue(platform.isPlaceholder());

        url = createPropertyFile("version=1.8.0\nclassifier=cu111-linux-x86_64");
        platform = Platform.fromUrl(url);
        // cu111 can run on cu113 machine
        Assert.assertTrue(platform.matches(system));
        // cu113 cannot run on cu111 machine (the same major version)
        Assert.assertFalse(system.matches(platform));

        url = createPropertyFile("version=1.8.0\nclassifier=cu102-linux-x86_64");
        platform = Platform.fromUrl(url);
        // cu102 (lower major version) can run on cu113 machine,
        Assert.assertTrue(platform.matches(system));
        // cu113 can not run on cu102 machine
        Assert.assertFalse(system.matches(platform));

        // MXNet
        url = createPropertyFile("version=1.8.0\nclassifier=cu113mkl-linux-x86_64");
        platform = Platform.fromUrl(url);
        Assert.assertTrue(platform.matches(system));

        // GPU machine should compatible with CPU package
        url = createPropertyFile("version=1.8.0\nclassifier=mkl-linux-x86_64");
        platform = Platform.fromUrl(url);
        Assert.assertTrue(platform.matches(system));

        url = createPropertyFile("version=1.8.0\nclassifier=cpu-linux-x86_64");
        platform = Platform.fromUrl(url);
        Assert.assertTrue(platform.matches(system));

        url = createPropertyFile("version=1.8.0\nclassifier=cpu-mac-x86_64");
        platform = Platform.fromUrl(url);
        Assert.assertFalse(platform.matches(system));

        url = createPropertyFile("version=1.8.0\nclassifier=cpu-linux-aar64");
        platform = Platform.fromUrl(url);
        Assert.assertFalse(platform.matches(system));
    }

    @Test
    public void testDetectPlatform() throws IOException, ReflectiveOperationException {
        Path dir = Paths.get("build/tmp/");
        Files.createDirectories(dir);
        Platform system = Platform.fromSystem();
        String classifier = system.getClassifier();
        createZipFile(0, "1.0", "cpu", classifier, true);
        createZipFile(1, "1.0", "cpu", classifier, false);
        createZipFile(2, "1.0", "cpu-precxx11", classifier, false);
        createZipFile(3, "1.0", "cu117", classifier, false);
        createZipFile(4, "1.0", "cu117-precxx11", classifier, false);
        createZipFile(5, "1.0", "cu999", classifier, false);
        createZipFile(6, "1.0", "cu999-precxx11", classifier, false);
        createZipFile(7, "99.99", "cu999-precxx11", classifier, false);
        System.setProperty("ai.djl.util.cuda.fork", "true");
        try {
            String[] gpuInfo = new String[] {"1", "99990", "90"};
            Field field = CudaUtils.class.getDeclaredField("gpuInfo");
            field.setAccessible(true);
            field.set(null, gpuInfo);
            URL[] urls = new URL[8];
            for (int i = 0; i < 8; ++i) {
                urls[i] = dir.resolve(i + ".jar").toUri().toURL();
            }
            URLClassLoader cl = new URLClassLoader(urls);
            Thread.currentThread().setContextClassLoader(cl);

            Platform detected = Platform.detectPlatform("pytorch");
            Assert.assertEquals(detected.getFlavor(), "cu999-precxx11");

            field.set(null, null);
        } finally {
            System.clearProperty("ai.djl.util.cuda.fork");
            Thread.currentThread().setContextClassLoader(null);
        }
    }

    private URL createPropertyFile(String content) throws IOException {
        Path dir = Paths.get("build/tmp/testFile/");
        Files.createDirectories(dir);
        Path file = dir.resolve("engine.properties");
        try (BufferedWriter writer = Files.newBufferedWriter(file)) {
            writer.append(content);
            writer.newLine();
        }
        return file.toUri().toURL();
    }

    private void createZipFile(
            int index, String version, String flavor, String classifier, boolean placeHolder)
            throws IOException {
        Path file = Paths.get("build/tmp/" + index + ".jar");
        try (JarOutputStream jos = new JarOutputStream(Files.newOutputStream(file))) {
            JarEntry entry = new JarEntry("native/lib/pytorch.properties");
            jos.putNextEntry(entry);
            if (placeHolder) {
                jos.write("placeholder=true\nversion=2.3.1".getBytes(StandardCharsets.UTF_8));
            } else {
                jos.write("version=".getBytes(StandardCharsets.UTF_8));
                jos.write(version.getBytes(StandardCharsets.UTF_8));
                jos.write("\nflavor=".getBytes(StandardCharsets.UTF_8));
                jos.write(flavor.getBytes(StandardCharsets.UTF_8));
                jos.write("\nclassifier=".getBytes(StandardCharsets.UTF_8));
                jos.write(classifier.getBytes(StandardCharsets.UTF_8));
            }
        }
    }
}
