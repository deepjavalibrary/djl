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
package ai.djl.tensorflow.engine;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class TfLabelMap {

    public static Map<Integer, String> parse(Path pbtxtPath) throws IOException {
        List<String> lines = Files.readAllLines(pbtxtPath);
        List<String> items = new ArrayList<>();
        items.add("");
        lines.forEach(line -> {
            if (line.contains("}")) {
                String item = (items.get(0) + line.substring(0, line.indexOf("}") + 1)).replaceAll("( )+", " ");
                item = item.substring(item.indexOf("{") + 1, item.indexOf("}")).trim();
                items.set(0, item);
                items.add(0, line.substring(line.indexOf("}") + 1));
            } else {
                items.set(0, items.get(0) + line);
            }
        });

        Map<Integer, String> map = new TreeMap<>();

        items.forEach(s -> {
            Scanner scanner = new Scanner(s);
            Integer id = null;
            String name = null;
            StringBuilder displayName = null;
            while (scanner.hasNext()) {
                String key = scanner.next().trim();
                key = key.replaceAll(":", "");
                switch (key) {
                    case "id":
                        id = scanner.nextInt();
                        break;
                    case "name":
                        name = scanner.next().trim();
                        break;
                    case "display_name":
                        displayName = new StringBuilder(scanner.next().trim());
                        if (displayName.charAt(0) == '"') {
                            while (displayName.charAt(displayName.length() - 1) != '"') {
                                displayName.append(" ").append(scanner.next().trim());
                            }
                            displayName = new StringBuilder(displayName.substring(1, displayName.length() - 1));
                        }
                        break;
                }
            }
            if (id != null)
                map.put(id, displayName.toString());
        });
        return map;
    }

    public static List<String> toSynset(Map<Integer, String> map) {
        List<String> list = new ArrayList<>();
        for (int i = 0; i <= Collections.max(map.keySet()); i++)
            list.add(map.getOrDefault(i, ""));
        return list;
    }
}




