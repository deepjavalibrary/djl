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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PairListTest {

    @Test
    public void testConstruction() {
        PairList<String, String> pairs =
                new PairList<>(Arrays.asList("Hello", "world"), Arrays.asList("Hello", "World"));
        Assert.assertEquals(pairs.size(), 2);
        Map<String, String> map = new HashMap<>();
        map.put("Hello", "world");
        map.put("ni", "hao");
        pairs = new PairList<>(map);
        Assert.assertEquals(pairs.size(), 2);
        Assert.assertEquals(pairs.keyAt(0), "Hello");
        Assert.assertEquals(pairs.keys().size(), 2);
        Assert.assertEquals(pairs.values().size(), 2);
    }

    @Test
    public void testToMap() {
        List<String> keys = new ArrayList<>();
        keys.add("Hello");
        keys.add("Hello");
        List<String> values = new ArrayList<>();
        values.add("Hello");
        values.add("Hello");
        values.add("Hello");
        Assert.assertThrows(IllegalArgumentException.class, () -> new PairList<>(keys, values));

        values.remove(0);
        PairList<String, String> pairs = new PairList<>(keys, values);
        String[] strArr = pairs.keyArray(new String[3]);
        Assert.assertEquals(strArr.length, 3);
        Assert.assertThrows(IllegalStateException.class, pairs::toMap);

        pairs.remove("Hello");
        Assert.assertEquals(pairs.size(), 1);
        pairs.remove("Hello");
        pairs.add("Hello", "World");
        Assert.assertEquals(pairs.toMap().size(), 1);
        Assert.assertEquals(pairs.get(0), new Pair<>("Hello", "World"));
    }

    @Test
    public void testRemove() {
        PairList<String, String> pairs = new PairList<>();
        String value = pairs.remove("not_found");
        Assert.assertNull(value);
    }

    @Test(expectedExceptions = NoSuchElementException.class)
    public void testIteratorException() {
        PairList<String, String> pairs = new PairList<>();
        pairs.iterator().next();
    }
}
