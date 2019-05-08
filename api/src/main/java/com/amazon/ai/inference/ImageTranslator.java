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
package com.amazon.ai.inference;

import com.amazon.ai.Translator;
import com.amazon.ai.image.Images;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;

public abstract class ImageTranslator<T> implements Translator<BufferedImage, T> {

    @Override
    public NDList processInput(Predictor<?, ?> predictor, BufferedImage input) {
        int w = input.getWidth();
        int h = input.getHeight();
        Shape shape = new Shape(1, 3, h, w);
        DataDesc dataDesc = new DataDesc(shape);

        FloatBuffer buffer = Images.toFloatBuffer(input);

        NDArray array = predictor.create(dataDesc);
        array.set(buffer);

        return new NDList(normalize(array));
    }

    protected NDArray normalize(NDArray array) {
        return array;
    }
}
