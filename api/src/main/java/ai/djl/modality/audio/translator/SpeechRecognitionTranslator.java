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

package ai.djl.modality.audio.translator;

import ai.djl.modality.audio.Audio;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/**
 * A {@link Translator} that post-process the {@link Audio} into {@link String} to get a text
 * translation of the audio.
 */
public class SpeechRecognitionTranslator implements NoBatchifyTranslator<Audio, String> {

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Audio input) throws Exception {
        float[] data = input.getData();
        NDArray array = ctx.getNDManager().create(data, new Shape(1, data.length));
        return new NDList(array);
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) throws Exception {
        return list.get(0).toStringArray()[0];
    }
}
