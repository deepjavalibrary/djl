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

import java.io.BufferedWriter;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

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
        Platform system = Platform.fromUrl(url);
        Assert.assertEquals(system.getFlavor(), "cu113");
        Assert.assertEquals(system.getClassifier(), "linux-x86_64");
        Assert.assertEquals(system.getOsPrefix(), "linux");
        Assert.assertEquals(system.getOsArch(), "x86_64");

        url = createPropertyFile("version=1.8.0\nplaceholder=true");
        Platform platform = Platform.fromUrl(url);
        Assert.assertTrue(platform.isPlaceholder());

        url = createPropertyFile("version=1.8.0\nclassifier=cu111-linux-x86_64");
        platform = Platform.fromUrl(url);
        Assert.assertFalse(platform.matches(system));
        Assert.assertTrue(platform.matches(system, false));

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
}
