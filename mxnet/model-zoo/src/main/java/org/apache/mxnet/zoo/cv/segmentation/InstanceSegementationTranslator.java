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
package org.apache.mxnet.zoo.cv.segmentation;

import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.translate.Translator;
import software.amazon.ai.translate.TranslatorContext;

public class InstanceSegementationTranslator implements Translator<NDList, NDList> {
    @Override
    public NDList processOutput(TranslatorContext ctx, NDList list) throws Exception {
        return null;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, NDList input) throws Exception {
        return null;
    }
}
