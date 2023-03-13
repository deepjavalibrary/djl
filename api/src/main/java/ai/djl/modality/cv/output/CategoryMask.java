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

    private static final int COLOR_BLACK = 0xFF000000;

    private static final Gson GSON =
            JsonUtils.builder()
                    .registerTypeAdapter(CategoryMask.class, new SegmentationSerializer())
                    .create();

    private List<String> classes;
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
     * Extracts the detected objects from the image.
     *
     * @param image the original image
     * @return the detected objects from the image
     */
    public Image getMaskImage(Image image) {
        return image.getMask(mask);
    }

    /**
     * Extracts the specified object from the image.
     *
     * @param image the original image
     * @param classId the class to extract from the image
     * @return the specific object on the image
     */
    public Image getMaskImage(Image image, int classId) {
        int width = mask[0].length;
        int height = mask.length;
        int[][] selected = new int[height][width];
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                selected[h][w] = mask[h][w] == classId ? 1 : 0;
            }
        }
        return image.getMask(selected);
    }

    /**
     * Extracts the background from the image.
     *
     * @param image the original image
     * @return the background of the image
     */
    public Image getBackgroundImage(Image image) {
        return getMaskImage(image, 0);
    }

    /**
     * Highlights the detected object on the image with random colors.
     *
     * @param image the original image
     * @param opacity the opacity of the overlay. Value is between 0 and 255 inclusive, where 0
     *     means the overlay is completely transparent and 255 means the overlay is completely
     *     opaque.
     */
    public void drawMask(Image image, int opacity) {
        drawMask(image, opacity, COLOR_BLACK);
    }

    /**
     * Highlights the detected object on the image with random colors.
     *
     * @param image the original image
     * @param opacity the opacity of the overlay. Value is between 0 and 255 inclusive, where 0
     *     means the overlay is completely transparent and 255 means the overlay is completely
     *     opaque.
     * @param background replace the background with specified background color, use transparent
     *     color to remove background
     */
    public void drawMask(Image image, int opacity, int background) {
        int[] colors = generateColors(background, opacity);
        Image maskImage = getColorOverlay(colors);
        image.drawImage(maskImage, true);
    }

    /**
     * Highlights the specified object with specific color.
     *
     * @param image the original image
     * @param classId the class to draw on the image
     * @param color the rgb color with opacity
     * @param opacity the opacity of the overlay. Value is between 0 and 255 inclusive, where 0
     *     means the overlay is completely transparent and 255 means the overlay is completely
     *     opaque.
     */
    public void drawMask(Image image, int classId, int color, int opacity) {
        int[] colors = new int[classes.size()];
        colors[classId] = color & 0xFFFFFF | opacity << 24;
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

    private int[] generateColors(int background, int opacity) {
        int[] colors = new int[classes.size()];
        colors[0] = background;
        for (int i = 1; i < classes.size(); i++) {
            int red = RandomUtils.nextInt(256);
            int green = RandomUtils.nextInt(256);
            int blue = RandomUtils.nextInt(256);
            colors[i] = opacity << 24 | red << 16 | green << 8 | blue;
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
