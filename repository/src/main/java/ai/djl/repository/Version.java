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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

public class Version implements Comparable<Version> {

    private String version;
    private boolean snapshot;
    private List<Object> comparable;

    public Version(String version) {
        this.version = version;
        comparable = new ArrayList<>();
        String[] parts = version.split("\\.", 5);
        int length = parts.length;
        snapshot = parts[length - 1].endsWith("-SNAPSHOT");
        if (snapshot) {
            parts[length - 1] = parts[length - 1].replaceAll("-SNAPSHOT", "");
        }
        for (String part : parts) {
            Integer value = tryParseInt(part);
            if (value != null) {
                comparable.add(value);
            } else {
                comparable.add(part);
            }
        }
    }

    @Override
    public int compareTo(Version otherVersion) {
        Comp comp = new Comp();
        List<Object> other = otherVersion.comparable;
        int currentSize = comparable.size();
        int otherSize = other.size();
        int size = Math.min(currentSize, otherSize);
        for (int i = 0; i < size; ++i) {
            int ret = comp.compare(comparable.get(i), other.get(i));
            if (ret != 0) {
                return ret;
            }
        }
        return Integer.compare(currentSize, otherSize);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        return compareTo((Version) o) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(comparable);
    }

    public int getMajorVersion() {
        return get(0);
    }

    public int getMinorVersion() {
        return get(1);
    }

    public int getIncrementalVersion() {
        return get(2);
    }

    public boolean isSnapshot() {
        return snapshot;
    }

    private int get(int index) {
        if (comparable.size() > index) {
            Object c = comparable.get(index);
            if (c instanceof Integer) {
                return (Integer) c;
            }
        }
        return 0;
    }

    private static Integer tryParseInt(String s) {
        try {
            long longValue = Long.parseLong(s);
            if (longValue > Integer.MAX_VALUE) {
                return null;
            }
            return (int) longValue;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    @Override
    public String toString() {
        return version;
    }

    private static final class Comp implements Comparator<Object>, Serializable {

        private static final long serialVersionUID = 1L;

        @Override
        public int compare(Object o1, Object o2) {
            if (o1 instanceof Integer) {
                if (o2 instanceof Integer) {
                    return ((Integer) o1).compareTo((Integer) o2);
                }
                return -1;
            }

            if (o2 instanceof Integer) {
                return 1;
            }
            return ((String) o1).compareTo((String) o2);
        }
    }
}
