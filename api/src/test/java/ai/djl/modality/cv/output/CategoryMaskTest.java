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
package ai.djl.modality.cv.output;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

public class CategoryMaskTest {

    private static final Logger logger = LoggerFactory.getLogger(CategoryMaskTest.class);

    @Test
    public void test() {
        List<String> classes = Arrays.asList("cat", "dog");
        CategoryMask mask = new CategoryMask(classes, new int[][] {{1}, {2}});

        logger.info("CategoryMask: {}", mask);
        Assert.assertEquals(mask.toJson(), "{\"classes\":[\"cat\",\"dog\"],\"mask\":[[1],[2]]}");
    }
}
