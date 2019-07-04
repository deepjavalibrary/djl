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
package software.amazon.ai.ndarray.types;

/**
 * An enum represents Sparse matrix storage formats.
 *
 * <ul>
 *   <li>DEFAULT: No sparse format
 *   <li>ROW_SPARSE: Row Sparse
 *   <li>CSR: Compressed Sparse Row
 * </ul>
 *
 * @see <a href="https://software.intel.com/en-us/node/471374">Sparse Matrix Storage Formats</a>
 */
public enum SparseFormat {
    UNDEFINED("undefined", -1),
    DEFAULT("default", 0),
    ROW_SPARSE("row_sparse", 1),
    CSR("csr", 2);

    private String type;
    private int value;

    SparseFormat(String type, int value) {
        this.type = type;
        this.value = value;
    }

    /**
     * Gets the {@code SparseFormat} from it's integer value.
     *
     * @param value integer value of the {@code SparseFormat}
     * @return {@code SparseFormat}
     */
    public static SparseFormat fromValue(int value) {
        for (SparseFormat t : values()) {
            if (value == t.getValue()) {
                return t;
            }
        }
        throw new IllegalArgumentException("Unknown storage type: " + value);
    }

    /**
     * Returns {@code SparseFormat} name.
     *
     * @return {@code SparseFormat} name.
     */
    public String getType() {
        return type;
    }

    /**
     * Returns integer value of this {@code SparseFormat}.
     *
     * @return integer value of this {@code SparseFormat}
     */
    public int getValue() {
        return value;
    }
}
