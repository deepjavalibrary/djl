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
package ai.djl.examples.inference.biggan;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class BigGANCategory {
    private static final Logger logger = LoggerFactory.getLogger(BigGANCategory.class);

    public static final int NUMBER_OF_CATEGORIES = 1000;
    private static final Map<String, BigGANCategory> CATEGORIES_BY_NAME =
            new ConcurrentHashMap<>(NUMBER_OF_CATEGORIES);
    private static String[] categoriesById;

    private int id;
    private String[] names;

    static {
        try {
            parseCategories();
        } catch (IOException e) {
            logger.error("Error parsing the ImageNet categories: {}", e);
        }
        createCategoriesByName();
    }

    private BigGANCategory(int id, String[] names) {
        this.id = id;
        this.names = names;
    }

    public int getId() {
        return id;
    }

    public String[] getNames() {
        return names.clone();
    }

    public static BigGANCategory id(int id) {
        String names = categoriesById[id];
        int index = names.indexOf(',');
        if (index < 0) {
            return of(names);
        } else {
            return of(names.substring(0, index));
        }
    }

    public static BigGANCategory of(String name) {
        if (!CATEGORIES_BY_NAME.containsKey(name)) {
            throw new IllegalArgumentException(name + " is not a valid category.");
        }
        return CATEGORIES_BY_NAME.get(name);
    }

    private static void createCategoriesByName() {
        for (int i = 0; i < NUMBER_OF_CATEGORIES; i++) {
            String[] categoryNames = categoriesById[i].split(", ");
            BigGANCategory category = new BigGANCategory(i, categoryNames);

            for (String name : categoryNames) {
                CATEGORIES_BY_NAME.put(name, category);
            }
        }
    }

    private static void parseCategories() throws IOException {
        String filePath = "src/main/resources/categories.txt";

        List<String> fileLines = Files.readAllLines(Paths.get(filePath));
        List<String> categories = new ArrayList<>(NUMBER_OF_CATEGORIES);
        for (String line : fileLines) {
            int nameIndex = line.indexOf(':') + 2;
            categories.add(line.substring(nameIndex));
        }

        categoriesById = categories.toArray(new String[] {});
    }
}
