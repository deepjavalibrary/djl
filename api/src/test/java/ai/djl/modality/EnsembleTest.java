/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class EnsembleTest {

    @Test
    public void testEnsembleClassifications() {
        List<String> classNames = Arrays.asList("a", "b", "c");
        List<Classifications> list = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            double prob = i;
            List<Double> probs = Arrays.asList(prob, prob, prob);
            list.add(new Classifications(classNames, probs));
        }
        Classifications actual = list.get(0).ensemble(list);
        Assert.assertEquals(actual.getClassNames(), classNames);
        Assert.assertEquals(actual.getProbabilities(), Arrays.asList(1.0, 1.0, 1.0));
    }
}
