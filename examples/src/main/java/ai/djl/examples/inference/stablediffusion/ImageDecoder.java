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
package ai.djl.examples.inference.stablediffusion;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

public class ImageDecoder implements NoBatchifyTranslator<NDArray, Image> {

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, NDArray input) throws Exception {
        input = input.div(0.18215);
        return new NDList(input);
    }

    /** {@inheritDoc} */
    @Override
    public Image processOutput(TranslatorContext ctx, NDList output) throws Exception {
        NDArray scaled = output.get(0).div(2).add(0.5).clip(0, 1);
        scaled = scaled.transpose(0, 2, 3, 1);
        scaled = scaled.mul(255).round().toType(DataType.INT8, true).get(0);
        return ImageFactory.getInstance().fromNDArray(scaled);
    }
}
