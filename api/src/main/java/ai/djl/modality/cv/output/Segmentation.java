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
package ai.djl.modality.cv.output;

import ai.djl.util.JsonSerializable;
import ai.djl.util.JsonUtils;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;

import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * A class representing the segmentation result of an image in an {@link
 * ai.djl.Application.CV#SEMANTIC_SEGMENTATION} case.
 */
public class Segmentation implements JsonSerializable {

    private static final long serialVersionUID = 1L;

    private static final Gson GSON =
            JsonUtils.builder()
                    .registerTypeAdapter(Segmentation.class, new SegmentationSerializer())
                    .create();

    private int[][] mask;

    /**
     * Constructs a Mask with the given data.
     *
     * @param mask the category mask for each pixel in the image
     */
    public Segmentation(int[][] mask) {
        this.mask = mask;
    }

    /**
     * Returns the class for each pixel.
     *
     * @return the class for each pixel
     */
    public int[][] getMask() {
        return mask;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap(toJson().getBytes(StandardCharsets.UTF_8));
    }

    /** {@inheritDoc} */
    @Override
    public String toJson() {
        return GSON.toJson(this) + '\n';
    }

    /** A customized Gson serializer to serialize the {@code Segmentation} object. */
    public static final class SegmentationSerializer implements JsonSerializer<Segmentation> {

        /** {@inheritDoc} */
        @Override
        public JsonElement serialize(Segmentation src, Type type, JsonSerializationContext ctx) {
            int[][] m = src.getMask();
            return ctx.serialize(m);
        }
    }
}
