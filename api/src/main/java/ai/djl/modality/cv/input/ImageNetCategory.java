/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.input;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A container for an ImageNet class. */
public final class ImageNetCategory {
    private static final Logger logger = LoggerFactory.getLogger(ImageNetCategory.class);

    public static final int NUMBER_OF_CATEGORIES = 1000;
    private static final Map<String, ImageNetCategory> CATEGORIES_BY_NAME =
            new ConcurrentHashMap<>(NUMBER_OF_CATEGORIES);
    private static String[] categoriesById;

    private int id;
    private String[] names;

    static {
        try {
            parseCategories();
        } catch (IOException e) {
            logger.error("Error parsing the ImageNet categories:", e);
        }
        createCategoriesByName();
    }

    private ImageNetCategory(int id, String[] names) {
        this.id = id;
        this.names = names;
    }

    /**
     * Returns the category ID.
     *
     * @return the category ID
     */
    public int getId() {
        return id;
    }

    /**
     * Returns the list of different names for this category.
     *
     * @return the list of different names for this category
     */
    public String[] getNames() {
        return names.clone();
    }

    /**
     * Get the ImageNet category with the ID.
     *
     * @param id of the category
     * @return the corresponding {@link ImageNetCategory}
     */
    public static ImageNetCategory id(int id) {
        String names = categoriesById[id];
        int index = names.indexOf(',');
        if (index < 0) {
            return of(names);
        } else {
            return of(names.substring(0, index));
        }
    }

    /**
     * Get the ImageNet category with one of its names.
     *
     * @param name of the category
     * @return the corresponding {@link ImageNetCategory}
     */
    public static ImageNetCategory of(String name) {
        if (!CATEGORIES_BY_NAME.containsKey(name)) {
            throw new IllegalArgumentException(name + " is not a valid category.");
        }
        return CATEGORIES_BY_NAME.get(name);
    }

    /** Create structure to map from ImageNet label to the corresponding ImageNetCategory object. */
    private static void createCategoriesByName() {
        for (int i = 0; i < NUMBER_OF_CATEGORIES; i++) {
            String[] categoryNames = categoriesById[i].split(", ");
            ImageNetCategory category = new ImageNetCategory(i, categoryNames);

            for (String name : categoryNames) {
                CATEGORIES_BY_NAME.put(name, category);
            }
        }
    }

    /**
     * Parse all the ImageNet labels from a text file and store them in an array where the index is
     * the ID.
     *
     * @throws IOException if an I/O error occurs reading from the file or a malformed or unmappable
     *     byte sequence is read
     */
    private static void parseCategories() throws IOException {
        String filePath =
                "../model-zoo/src/test/resources/mlrepo/model/cv/image_classification/ai/djl/zoo/synset_imagenet.txt";

        List<String> fileLines = Files.readAllLines(Paths.get(filePath));
        List<String> categories = new ArrayList<>(NUMBER_OF_CATEGORIES);
        for (String line : fileLines) {
            categories.add(line.substring(10));
        }

        categoriesById = categories.toArray(new String[] {});
    }
}
