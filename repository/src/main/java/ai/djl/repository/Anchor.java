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
package ai.djl.repository;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Anchor {

    private String[] path;

    public Anchor(String... path) {
        this.path = path;
    }

    public static Anchor parse(String anchor) {
        String[] tokens = anchor.split("[:/]");
        return new Anchor(tokens);
    }

    public Anchor normalize() {
        List<String> parts = new ArrayList<>();
        for (String s : path) {
            String[] tokens = s.split("/");
            Collections.addAll(parts, tokens);
        }
        return new Anchor(parts.toArray(new String[0]));
    }

    public String get(int index) {
        return path[index];
    }

    public String getPath() {
        return String.join("/", path);
    }

    public Anchor resolve(Anchor other) {
        String[] newPath = new String[path.length + other.path.length];
        System.arraycopy(path, 0, newPath, 0, path.length);
        System.arraycopy(other.path, 0, newPath, path.length, other.path.length);
        return new Anchor(newPath);
    }

    public Anchor resolve(String... others) {
        Anchor anchor = this;
        for (String other : others) {
            anchor = anchor.resolve(Anchor.parse(other));
        }
        return anchor;
    }
}
