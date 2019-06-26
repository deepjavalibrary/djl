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
package software.amazon.ai.inference;

import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import software.amazon.ai.Translator;
import software.amazon.ai.TranslatorContext;
import software.amazon.ai.image.Images;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;

/**
 * Builtin <code>Translator</code> that provide default pre-process image.
 *
 * @param <T> output object type
 */
public abstract class ImageTranslator<T> implements Translator<BufferedImage, T> {

    /**
     * Process the <code>BufferedImage</code> input and convert to NDList.
     *
     * @param ctx toolkit that would help to creating input NDArray
     * @param input <code>BufferedImage</code> input
     * @return {@link NDList}
     */
    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage input) {
        int w = input.getWidth();
        int h = input.getHeight();
        Shape shape = new Shape(1, 3, h, w);
        DataDesc dataDesc = new DataDesc(shape);

        FloatBuffer buffer = Images.toFloatBuffer(input);

        NDArray array = ctx.getNDFactory().create(dataDesc);
        array.set(buffer);

        return new NDList(normalize(array));
    }

    /**
     * Normalize pre-processed {@link NDArray}.
     *
     * <p>It's expected that developer to override this method to provide customized normalization.
     *
     * @param array pre-processed {@link NDArray}
     * @return normalized NDArray
     */
    protected NDArray normalize(NDArray array) {
        return array;
    }
}
