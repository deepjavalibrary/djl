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

/**
 * An {@code Anchor} represents a multi-level category of {@link Metadata} in a {@link MRL}.
 *
 * <p>The paths can have subpaths separated by slashes such as "dataset/cv" and "dataset/nlp". The
 * anchors translate to directories. Directories sharing a path prefix can be used to organize a
 * multi-level hierarchy of categories.
 *
 * <p>Some example paths are located in {@link MRL.Dataset} and {@link MRL.Model}.
 *
 * @see MRL
 */
public class Anchor {

    private String[] path;

    /**
     * Constructs an anchor from a split path.
     *
     * @param path a split path where each element in the path corresponds to a directory
     */
    public Anchor(String... path) {
        this.path = path;
    }

    /**
     * Creates an anchor from a file path string.
     *
     * @param anchor the string containing each level separated by "/"
     * @return the new anchor
     */
    public static Anchor parse(String anchor) {
        String[] tokens = anchor.split("[:/]");
        return new Anchor(tokens);
    }

    /**
     * Splits path elements that contain multiple levels into separate components of the path.
     *
     * <p>For example, it will convert path("a/b","c","d/e/f") to path("a", "b", "c", "d", "e",
     * "f").
     *
     * @return a new split anchor
     */
    public Anchor normalize() {
        List<String> parts = new ArrayList<>();
        for (String s : path) {
            String[] tokens = s.split("/");
            Collections.addAll(parts, tokens);
        }
        return new Anchor(parts.toArray(new String[0]));
    }

    /**
     * Returns the path element at the given index.
     *
     * @param index the index to retrieve
     * @return the path element at the given index
     */
    public String get(int index) {
        return path[index];
    }

    /**
     * Returns the path as a single "/" separated string.
     *
     * @return the path as a single "/" separated string
     */
    public String getPath() {
        return String.join("/", path);
    }

    /**
     * Joins two anchors together.
     *
     * <p>When joined, this this.path is the prefix and other.path is the suffix of the resulting
     * path.
     *
     * @param other the path to append
     * @return the joined path
     */
    public Anchor resolve(Anchor other) {
        String[] newPath = new String[path.length + other.path.length];
        System.arraycopy(path, 0, newPath, 0, path.length);
        System.arraycopy(other.path, 0, newPath, path.length, other.path.length);
        return new Anchor(newPath);
    }

    /**
     * Appends path items to the anchor.
     *
     * @param others the path elements to append
     * @return this anchor
     */
    public Anchor resolve(String... others) {
        Anchor anchor = this;
        for (String other : others) {
            anchor = anchor.resolve(Anchor.parse(other));
        }
        return anchor;
    }
}
