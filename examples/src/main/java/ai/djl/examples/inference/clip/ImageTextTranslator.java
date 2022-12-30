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
package ai.djl.examples.inference.clip;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;

public class ImageTextTranslator implements NoBatchifyTranslator<Pair<Image, String>, float[]> {

    private ImageTranslator imgTranslator;
    private TextTranslator txtTranslator;

    public ImageTextTranslator() {
        this.imgTranslator = new ImageTranslator();
        this.txtTranslator = new TextTranslator();
    }

    /** {@inheritDoc} */
    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDArray logitsPerImage = list.get(0);
        return logitsPerImage.toFloatArray();
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Pair<Image, String> input) throws Exception {
        NDList imageInput = imgTranslator.processInput(ctx, input.getKey());
        NDList textInput = txtTranslator.processInput(ctx, input.getValue());
        return new NDList(textInput.get(0), imageInput.get(0), textInput.get(1));
    }
}
