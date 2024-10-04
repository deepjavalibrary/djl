/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import org.apache.commons.codec.binary.Base64OutputStream;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

/** A {@link Translator} that can serve SAM2 model. */
public class Sam2ServingTranslator implements Translator<Input, Output> {

    private Sam2Translator translator;

    /**
     * Constructs a new {@code Sam2ServingTranslator} instance.
     *
     * @param translator a {@code Sam2Translator}
     */
    public Sam2ServingTranslator(Sam2Translator translator) {
        this.translator = translator;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return translator.getBatchifier();
    }

    /** {@inheritDoc} */
    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) throws IOException {
        Output output = new Output();
        Sam2Input sam2 = (Sam2Input) ctx.getAttachment("input");
        output.addProperty("Content-Type", "application/json");
        DetectedObjects detection = translator.processOutput(ctx, list);
        Map<String, Object> ret = new LinkedHashMap<>(); // NOPMD
        ret.put("result", detection);
        if (sam2.isVisualize()) {
            Image img = sam2.getImage();
            img.drawBoundingBoxes(detection, 0.8f);
            img.drawMarks(sam2.getPoints());
            for (Rectangle rect : sam2.getBoxes()) {
                img.drawRectangle(rect, 0xff0000, 6);
            }
            ByteArrayOutputStream os = new ByteArrayOutputStream();
            os.write("data:image/png;base64,".getBytes(StandardCharsets.UTF_8));
            Base64OutputStream bos = new Base64OutputStream(os, true, 0, null);
            img.save(bos, "png");
            bos.close();
            os.close();
            ret.put("image", os.toString(StandardCharsets.UTF_8.name()));
        }
        output.add(BytesSupplier.wrapAsJson(ret));
        return output;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
        BytesSupplier data = input.getData();
        try {
            if (data == null) {
                throw new TranslateException("Input data is empty.");
            }
            Sam2Input sam2 = Sam2Input.fromJson(data.getAsString());
            ctx.setAttachment("input", sam2);
            return translator.processInput(ctx, sam2);
        } catch (IOException e) {
            throw new TranslateException("Input is not an Image data type", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        translator.prepare(ctx);
    }
}
