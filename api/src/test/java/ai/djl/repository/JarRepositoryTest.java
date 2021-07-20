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
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class JarRepositoryTest {

    @Test
    public void testResource() throws IOException {
        Path path = Paths.get("build/resources/test/");
        Files.createDirectories(path);

        Path dir = Paths.get("build/testDir/");
        Utils.deleteQuietly(dir);
        Files.createDirectories(dir);
        Files.createFile(dir.resolve("synset.txt"));
        Path testFile = path.resolve("test.zip");
        ZipUtils.zip(dir, testFile, false);

        Repository repo = Repository.newInstance("test", "jar:///test.zip");
        Assert.assertEquals("test", repo.getName());
        Assert.assertTrue(repo.isRemote());

        List<MRL> list = repo.getResources();
        Assert.assertEquals(list.size(), 1);

        Artifact artifact = repo.resolve(list.get(0), null);
        repo.prepare(artifact);
        Assert.assertEquals(1, artifact.getFiles().size());
    }
}
