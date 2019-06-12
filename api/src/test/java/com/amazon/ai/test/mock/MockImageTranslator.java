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
package com.amazon.ai.test.mock;

import com.amazon.ai.TranslatorContext;
import com.amazon.ai.image.Rectangle;
import com.amazon.ai.inference.DetectedObject;
import com.amazon.ai.inference.ImageTranslator;
import com.amazon.ai.ndarray.NDList;
import java.util.ArrayList;
import java.util.List;

public class MockImageTranslator extends ImageTranslator<List<DetectedObject>> {

    private List<DetectedObject> output;

    public MockImageTranslator(String className) {
        output = new ArrayList<>(1);
        output.add(new DetectedObject(className, 0.8, new Rectangle(0, 0, 1, 1)));
    }

    @Override
    public List<DetectedObject> processOutput(TranslatorContext ctx, NDList list) {
        return output;
    }

    public void setOutput(List<DetectedObject> output) {
        this.output = output;
    }
}
