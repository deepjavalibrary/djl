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

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.util.JsonSerializable;
import ai.djl.util.JsonUtils;
import ai.djl.util.RandomUtils;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;

import java.awt.Color;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * A class representing the segmentation result of an image in an {@link
 * ai.djl.Application.CV#SEMANTIC_SEGMENTATION} case.
 */
public class CategoryMask implements JsonSerializable {

    private static final long serialVersionUID = 1L;

    private static final Gson GSON =
            JsonUtils.builder()
                    .registerTypeAdapter(CategoryMask.class, new SegmentationSerializer())
                    .create();

    private transient List<String> classes;
    private int[][] mask;

    /**
     * Constructs a Mask with the given data.
     *
     * @param classes the list of classes
     * @param mask the category mask for each pixel in the image
     */
    public CategoryMask(List<String> classes, int[][] mask) {
        this.classes = classes;
        this.mask = mask;
    }

    /**
     * Returns the list of classes.
     *
     * @return list of classes
     */
    public List<String> getClasses() {
        return classes;
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

    /**
     * Highlights the detected object on the image with random colors.
     *
     * @param image the original image
     * @param transparency the transparency of the overlay
     */
    public void drawMask(Image image, int transparency) {
        drawMask(image, transparency, Color.BLACK.getRGB());
    }

    /**
     * Highlights the detected object on the image with random colors.
     *
     * @param image the original image
     * @param transparency the transparency of the overlay
     * @param background replace the background with specified background color, use transparent
     *     color to remove background
     */
    public void drawMask(Image image, int transparency, int background) {
        int[] colors = generateColors(background, transparency);
        Image maskImage = getColorOverlay(colors);
        image.drawImage(maskImage, true);
    }

    /**
     * Highlights the detected object on the image with random colors.
     *
     * @param image the original image
     * @param transparency the transparency of the overlay
     * @param background replace the background with specified image
     */
    public void drawMask(Image image, int transparency, Image background) {
        int[] colors = generateColors(0, transparency);
        Image colorOverlay = getColorOverlay(colors);
        Image maskImage = image.getMask(mask);
        image.drawImage(background, true);
        image.drawImage(maskImage, true);
        image.drawImage(colorOverlay, true);
    }

    /**
     * Highlights the specified object with specific color.
     *
     * @param image the original image
     * @param classId the class to draw on the image
     * @param color the rgb color with transparency
     * @param transparency the transparency of the overlay
     */
    public void drawMask(Image image, int classId, int color, int transparency) {
        int[] colors = new int[classes.size()];
        colors[classId] = color | transparency << 24;
        Image colorOverlay = getColorOverlay(colors);
        image.drawImage(colorOverlay, true);
    }

    private Image getColorOverlay(int[] colors) {
        int height = mask.length;
        int width = mask[0].length;
        int[] pixels = new int[width * height];
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int index = mask[h][w];
                pixels[h * width + w] = colors[index];
            }
        }
        return ImageFactory.getInstance().fromPixels(pixels, width, height);
    }

    private int[] generateColors(int background, int transparency) {
        int[] colors = new int[classes.size()];
        colors[0] = background;
        for (int i = 1; i < classes.size(); i++) {
            int red = RandomUtils.nextInt(256);
            int green = RandomUtils.nextInt(256);
            int blue = RandomUtils.nextInt(256);
            colors[i] = transparency << 24 | red << 16 | green << 8 | blue;
        }
        return colors;
    }

    /** A customized Gson serializer to serialize the {@code Segmentation} object. */
    public static final class SegmentationSerializer implements JsonSerializer<CategoryMask> {

        /** {@inheritDoc} */
        @Override
        public JsonElement serialize(CategoryMask src, Type type, JsonSerializationContext ctx) {
            return ctx.serialize(src.getMask());
        }
    }
}
