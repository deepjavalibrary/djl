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
package ai.djl.modality.nlp.preprocess;

import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TextEndpointTokenTest {

    @Test
    public void testDefaultEndpoints() {
        List<String> initial = Arrays.asList("a", "b", "c");
        List<String> expected = Arrays.asList("<bos>", "a", "b", "c", "<eos>");
        TextProcessor processor = new TextTerminator();
        List<String> actual = processor.preprocess(initial);
        Assert.assertEquals(actual, expected);
    }

    @Test
    public void testDefaultStart() {
        List<String> initial = Arrays.asList("a", "b", "c");
        List<String> expected = Arrays.asList("<bos>", "a", "b", "c");
        TextProcessor processor = new TextTerminator(true, false);
        List<String> actual = processor.preprocess(initial);
        Assert.assertEquals(actual, expected);
    }

    @Test
    public void testCustomEnd() {
        List<String> initial = Arrays.asList("a", "b", "c");
        List<String> expected = Arrays.asList("a", "b", "c", "y");
        TextProcessor processor = new TextTerminator(false, true, "x", "y");
        List<String> actual = processor.preprocess(initial);
        Assert.assertEquals(actual, expected);
    }
}
