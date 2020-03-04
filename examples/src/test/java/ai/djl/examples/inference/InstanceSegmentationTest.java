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
package ai.djl.examples.inference;

import ai.djl.ModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class InstanceSegmentationTest {

    @Test
    public void testInstanceSegmentation() throws ModelException, TranslateException, IOException {
        DetectedObjects result = InstanceSegmentation.predict();
        Classifications.Classification best = result.best();
        Assert.assertEquals(best.getClassName(), "bicycle");
        Assert.assertTrue(Double.compare(best.getProbability(), 0.9) > 0);
    }
}
