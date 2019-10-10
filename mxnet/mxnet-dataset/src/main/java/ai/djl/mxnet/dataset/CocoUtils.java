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

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CocoUtils {
    private Path annotationPath;
    private boolean prepared;
    private List<Long> imageIds;
    private Map<Long, CocoMetadata.Image> imageMap;
    private Map<Long, CocoMetadata.Annotation> annotationMap;
    private Map<Long, List<Long>> imageToAnn;
    private Map<Long, Integer> categoryIdMap;

    public CocoUtils(Path annotationPath) {
        this.annotationPath = annotationPath;
        imageIds = new ArrayList<>();
        imageMap = new HashMap<>();
        annotationMap = new HashMap<>();
        imageToAnn = new HashMap<>();
        categoryIdMap = new HashMap<>();
    }

    public void prepare() throws IOException {
        if (!prepared) {
            CocoMetadata metadata;
            try (Reader reader = Files.newBufferedReader(annotationPath)) {
                metadata = CocoMetadata.GSON.fromJson(reader, CocoMetadata.class);
            }
            createIndex(metadata);
            prepared = true;
        }
    }

    private void createIndex(CocoMetadata metadata) {
        for (CocoMetadata.Annotation annotation : metadata.getAnnotations()) {
            long imageId = annotation.getImageId();
            long id = annotation.getId();
            if (!imageToAnn.containsKey(imageId)) {
                imageToAnn.put(annotation.getImageId(), new ArrayList<>());
            }
            imageToAnn.get(imageId).add(id);
            annotationMap.put(id, annotation);
        }

        for (CocoMetadata.Image image : metadata.getImages()) {
            imageIds.add(image.getId());
            imageMap.put(image.getId(), image);
        }

        // create categoryIndex
        List<Long> categoryIds = new ArrayList<>();
        for (CocoMetadata.Category category : metadata.getCategories()) {
            categoryIds.add(category.getId());
        }
        for (int i = 0; i < categoryIds.size(); i++) {
            categoryIdMap.put(categoryIds.get(i), i);
        }
        // sort to keep the dataset ordered
        Collections.sort(imageIds);
    }

    public List<Long> getImageIds() {
        return imageIds;
    }

    public Path getRelativeImagePath(long imageId) {
        CocoMetadata.Image image = imageMap.get(imageId);
        String[] cocoUrl = image.getCocoUrl().split("/");
        return Paths.get(cocoUrl[cocoUrl.length - 2])
                .resolve(Paths.get(cocoUrl[cocoUrl.length - 1]));
    }

    public List<Long> getAnnotationIdByImageId(long imageId) {
        return imageToAnn.get(imageId);
    }

    public CocoMetadata.Annotation getAnnotationById(long annotationId) {
        return annotationMap.get(annotationId);
    }

    public int mapCategoryId(long originalCategoryId) {
        return categoryIdMap.get(originalCategoryId);
    }
}
