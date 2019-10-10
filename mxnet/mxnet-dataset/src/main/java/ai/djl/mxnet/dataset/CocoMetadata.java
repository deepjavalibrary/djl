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
package ai.djl.mxnet.dataset;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.annotations.SerializedName;
import java.lang.reflect.Type;
import java.util.List;
import software.amazon.ai.modality.cv.Rectangle;

public class CocoMetadata {

    public static final Gson GSON =
            new GsonBuilder()
                    .registerTypeAdapter(Rectangle.class, new RectangleDeserializer())
                    .create();

    private List<Image> images;
    private List<Annotation> annotations;
    private List<Category> categories;

    public List<Annotation> getAnnotations() {
        return annotations;
    }

    public List<Category> getCategories() {
        return categories;
    }

    public List<Image> getImages() {
        return images;
    }

    public static final class Annotation {

        @SerializedName("image_id")
        private long imageId;

        private long id;

        @SerializedName("bbox")
        private Rectangle bBox;

        private double area;

        @SerializedName("category_id")
        private long categoryId;

        public long getImageId() {
            return imageId;
        }

        public long getId() {
            return id;
        }

        public Rectangle getBoundingBox() {
            return bBox;
        }

        public long getCategoryId() {
            return categoryId;
        }

        public double getArea() {
            return area;
        }
    }

    public static final class Image {

        private int id;

        @SerializedName("coco_url")
        private String cocoUrl;

        private int height;
        private int width;

        public long getId() {
            return id;
        }

        public String getCocoUrl() {
            return cocoUrl;
        }

        public int getHeight() {
            return height;
        }

        public int getWidth() {
            return width;
        }
    }

    public static final class Category {

        private long id;

        public long getId() {
            return id;
        }
    }

    public static final class RectangleDeserializer implements JsonDeserializer<Rectangle> {

        @Override
        public Rectangle deserialize(
                JsonElement json, Type typeOfT, JsonDeserializationContext ctx) {
            JsonArray array = json.getAsJsonArray();
            return new Rectangle(
                    array.get(0).getAsDouble(),
                    array.get(1).getAsDouble(),
                    array.get(2).getAsDouble(),
                    array.get(3).getAsDouble());
        }
    }
}
