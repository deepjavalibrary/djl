/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.util.Utils;
import ai.djl.util.ZipUtils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class JarRepositoryTest {

    @Test
    public void testResource() throws IOException {
        Path path = Paths.get("build/tmp/");
        Files.createDirectories(path);

        Path dir = Paths.get("build/testDir/");
        Utils.deleteQuietly(dir);
        Files.createDirectories(dir);
        Files.createFile(dir.resolve("synset.txt"));
        Path testFile = path.resolve("test.zip");
        ZipUtils.zip(dir, testFile, false);
        Path jarFile = path.resolve("test.jar");
        ZipUtils.zip(testFile, jarFile, false);

        URL[] url = {jarFile.toUri().toURL()};
        try {
            Thread.currentThread().setContextClassLoader(new URLClassLoader(url));
            Repository repo = Repository.newInstance("test", "jar:///test.zip?hash=1");
            Assert.assertEquals("test", repo.getName());
            Assert.assertTrue(repo.isRemote());

            List<MRL> list = repo.getResources();
            Assert.assertEquals(list.size(), 1);

            Artifact artifact = repo.resolve(list.get(0), null);
            repo.prepare(artifact);
            Assert.assertEquals(1, artifact.getFiles().size());
        } finally {
            Thread.currentThread().setContextClassLoader(null);
        }
    }
}
