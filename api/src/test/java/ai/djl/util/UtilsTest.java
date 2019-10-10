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

package ai.djl.util;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
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
        String[] words = {"Hello World", "2"};
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
        Path dir = Paths.get("build/tmp/testFile/");
        Files.createDirectories(dir);
        Path file = dir.resolve("synset.txt");
        Assert.assertTrue(Utils.readLines(file).isEmpty());

        try (BufferedWriter writer = Files.newBufferedWriter(file)) {
            writer.append("line1");
            writer.newLine();
        }
        Assert.assertEquals(Utils.readLines(file).size(), 1);

        Utils.deleteQuietly(dir);
        Assert.assertTrue(Files.notExists(dir));
    }

    @Test
    public void testToFloatArray() {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(2);
        float[] array = Utils.toFloatArray(list);
        Assert.assertEquals(array, new float[] {1f, 2f});
    }
}
