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
package ai.djl.mxnet.integration;

import ai.djl.mxnet.engine.MxNDManager;
import ai.djl.mxnet.engine.Symbol;
import ai.djl.ndarray.NDManager;
import ai.djl.training.util.DownloadUtils;
import java.io.IOException;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class MxBackendOptimizationTest {

    @BeforeTest
    public void downloadSymbol() throws IOException {
        String url =
                "https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/mxnet/resnet/0.0.1/resnet18_v1-symbol.json";
        DownloadUtils.download(url, "build/symbol/resnet18_v1-symbol.json");
    }

    @Test
    public void testOptimizedFor() {
        // TODO: Add Customized plugin test
        try (MxNDManager manager = (MxNDManager) NDManager.newBaseManager()) {
            Symbol symbol = Symbol.load(manager, "build/symbol/resnet18_v1-symbol.json");
            symbol.optimizeFor("test", manager.getDevice());
        }
    }
}
