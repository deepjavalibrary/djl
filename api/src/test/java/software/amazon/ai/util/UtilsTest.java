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

package software.amazon.ai.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class UtilsTest {

    @BeforeClass
    public void preprocess() {
        Utils.deleteQuietly(Paths.get("build/tmp/testFile/"));
    }

    @AfterClass
    public void postprocess() {
        Utils.deleteQuietly(Paths.get("build/tmp/testFile/"));
    }

    @Test
    public void testContains() {
        String[] words = new String[] {"Hello World", "2"};
        Assert.assertTrue(Utils.contains(words, "2"));
        Assert.assertFalse(Utils.contains(words, "3"));
    }

    @Test
    public void testPad() {
        StringBuilder sb = new StringBuilder();
        sb.append("Hello");
        Utils.pad(sb, 'a', 5);
        Assert.assertEquals(sb.toString(), "Helloaaaaa");
    }

    @Test
    public void testFile() throws IOException {
        String dir = "build/tmp/testFile/";
        Files.createDirectories(Paths.get(dir));
        Files.createFile(Paths.get(dir + "synset.txt"));
        Utils.deleteQuietly(Paths.get(dir));
        Assert.assertTrue(Files.notExists(Paths.get(dir)));
        Assert.assertTrue(Utils.readLines(Paths.get(dir)).isEmpty());
        Files.createDirectories(Paths.get(dir));
        Files.createFile(Paths.get(dir + "synset.txt"));
        Assert.assertTrue(Utils.readLines(Paths.get(dir)).isEmpty());
    }
}
