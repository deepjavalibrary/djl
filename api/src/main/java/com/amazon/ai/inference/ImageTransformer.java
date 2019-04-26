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

import com.amazon.ai.Transformer;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.image.Images;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.types.DataDesc;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;

public abstract class ImageTransformer<T> implements Transformer<BufferedImage, T> {

    private DataDesc dataDesc;

    public ImageTransformer(DataDesc dataDesc) {
        this.dataDesc = dataDesc;
    }

    @Override
    public NDArray processInput(BufferedImage input) {
        FloatBuffer buffer = Images.toFloatBuffer(input);
        NDFactory factory = Engine.getInstance().getNDFactory();

        NDArray array = factory.create(dataDesc);
        array.set(buffer);

        return normalize(array);
    }

    protected NDArray normalize(NDArray array) {
        return array;
    }
}
