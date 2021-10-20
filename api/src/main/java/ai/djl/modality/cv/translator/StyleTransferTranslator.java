/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.translator;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

/** Built-in {@code Translator} that provides preprocessing and postprocessing for StyleTransfer. */
public class StyleTransferTranslator implements NoBatchifyTranslator<Image, Image> {

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray image = switchFormat(input.toNDArray(ctx.getNDManager())).expandDims(0);
        return new NDList(image.toType(DataType.FLOAT32, false));
    }

    /** {@inheritDoc} */
    @Override
    public Image processOutput(TranslatorContext ctx, NDList list) {
        NDArray output = list.get(0).addi(1).muli(128).toType(DataType.UINT8, false);
        return ImageFactory.getInstance().fromNDArray(output.squeeze());
    }

    private NDArray switchFormat(NDArray array) {
        return NDArrays.stack(array.split(3, 2)).squeeze();
    }
}
