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
package ai.djl.examples.inference.sr;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.Arrays;

public class SRTranslator implements Translator<Image[], Image[]> {

    @Override
    public NDList processInput(TranslatorContext ctx, Image[] input) throws Exception {
        NDManager manager = ctx.getNDManager();
        NDArray[] arrays =
                Arrays.stream(input).map(image -> image.toNDArray(manager)).toArray(NDArray[]::new);
        NDArray batch = NDArrays.stack(new NDList(arrays)).toType(DataType.FLOAT32, false);
        return new NDList(batch);
    }

    @Override
    public Image[] processOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDArray output = list.get(0).clip(0, 255).toType(DataType.UINT8, false);
        int sampleSize = (int) output.getShape().get(0);

        return output.split(sampleSize)
                .stream()
                .map(array -> ImageFactory.getInstance().fromNDArray(array.squeeze()))
                .toArray(Image[]::new);
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}
