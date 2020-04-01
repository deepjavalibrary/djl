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
package ai.djl.util;

/** {@code Hex} is a set of utilities for working with Hexadecimal Strings. */
public final class Hex {

    private static final char[] HEX_CHARS = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
    };

    private Hex() {}

    /**
     * Converts a byte array to a hex string.
     *
     * @param block the bytes to convert
     * @return the converted hex String
     */
    public static String toHexString(byte[] block) {
        if (block == null) {
            return null;
        }

        StringBuilder buf = new StringBuilder();
        for (byte aBlock : block) {
            int high = ((aBlock & 0xf0) >> 4);
            int low = (aBlock & 0x0f);

            buf.append(HEX_CHARS[high]);
            buf.append(HEX_CHARS[low]);
        }
        return buf.toString();
    }

    /**
     * Converts a hex string to a byte array.
     *
     * @param s the string to convert
     * @return the converted byte array
     */
    public static byte[] toByteArray(String s) {
        int len = s.length();

        if ((len % 2) != 0) {
            throw new NumberFormatException("Invalid Hex String");
        }

        byte[] ret = new byte[len / 2];
        for (int i = 0; i < len / 2; i++) {
            ret[i] = (byte) Integer.parseInt(s.substring(i * 2, i * 2 + 2), 16);
        }

        return ret;
    }
}
