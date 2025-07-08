/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.genai.gemini;

import ai.djl.testing.CoverageUtils;

import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URISyntaxException;

public class GenAiCoverageTest {

    @Test
    public void test() throws IOException, ReflectiveOperationException, URISyntaxException {
        CoverageUtils.testGetterSetters(GeminiInput.class);
    }
}
