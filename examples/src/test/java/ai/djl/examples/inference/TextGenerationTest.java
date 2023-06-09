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
package ai.djl.examples.inference;

import ai.djl.MalformedModelException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.testing.TestRequirements;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TextGenerationTest {

    @Test
    public void testTextGeneration()
            throws ModelNotFoundException, MalformedModelException, IOException {
        TestRequirements.engine("PyTorch");

        String[] args = new String[] {};

        // LMBlock
        Assert.assertEquals(LLMBlock.main(args), 0);

        // LMSearch
        AutoRegressiveSearch search = new AutoRegressiveSearch();
        Assert.assertTrue(search.mainContrastivePt(args));
        Assert.assertTrue(search.mainGreedyPt(args));
        Assert.assertTrue(search.mainBeamPt(args));
        Assert.assertTrue(search.mainBeamOnnx(args));

        // DynamicSequenceScheduler
        Assert.assertTrue(DynamicSequenceScheduler.mainContrastivePt());
    }
}
