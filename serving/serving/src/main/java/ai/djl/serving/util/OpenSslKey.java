/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.util;

/** A utility class converting OpenSSL private key to PKCS8 private key. */
public final class OpenSslKey {

    private static final int[] RSA_ENCRYPTION = {1, 2, 840, 113549, 1, 1, 1};
    private static final byte[] NULL_BYTES = {0x05, 0x00};

    private OpenSslKey() {}

    /**
     * Convert OpenSSL private key to PKCS8 private key.
     *
     * @param keySpec OpenSSL key spec
     * @return PKCS8 encoded private key
     */
    public static byte[] convertPrivateKey(byte[] keySpec) {
        if (keySpec == null) {
            return null;
        }

        byte[] bytes = new byte[keySpec.length];
        System.arraycopy(keySpec, 0, bytes, 0, keySpec.length);
        byte[] octetBytes = encodeOctetString(bytes);

        byte[] oidBytes = encodeOID(RSA_ENCRYPTION);
        byte[] verBytes = {0x02, 0x01, 0x00};

        byte[][] seqBytes = new byte[4][];
        seqBytes[0] = oidBytes;
        seqBytes[1] = NULL_BYTES;
        seqBytes[2] = null;

        byte[] oidSeqBytes = encodeSequence(seqBytes);

        seqBytes[0] = verBytes;
        seqBytes[1] = oidSeqBytes;
        seqBytes[2] = octetBytes;
        seqBytes[3] = null;

        return encodeSequence(seqBytes);
    }

    private static byte[] encodeOID(int[] oid) {
        if (oid == null) {
            return null;
        }

        int oLen = 1;

        for (int i = 2; i < oid.length; i++) {
            oLen += getOIDCompLength(oid[i]);
        }

        int len = oLen + getLengthOfLengthField(oLen) + 1;

        byte[] bytes = new byte[len];

        bytes[0] = 0x06; // ASN Object ID
        int offset = writeLengthField(bytes, oLen);

        bytes[offset++] = (byte) (40 * oid[0] + oid[1]);

        for (int i = 2; i < oid.length; i++) {
            offset = writeOIDComp(oid[i], bytes, offset);
        }

        return bytes;
    }

    private static byte[] encodeOctetString(byte[] bytes) {
        if (bytes == null) {
            return null;
        }

        int oLen = bytes.length; // one byte for unused bits field
        int len = oLen + getLengthOfLengthField(oLen) + 1;

        byte[] newBytes = new byte[len];

        newBytes[0] = 0x04;
        int offset = writeLengthField(newBytes, oLen);

        if (len - oLen != offset) {
            return null;
        }

        System.arraycopy(bytes, 0, newBytes, offset, oLen);
        return newBytes;
    }

    private static byte[] encodeSequence(byte[][] byteArrays) {
        if (byteArrays == null) {
            return null;
        }

        int oLen = 0;
        for (byte[] b : byteArrays) {
            if (b == null) {
                break;
            }

            oLen += b.length;
        }

        int len = oLen + getLengthOfLengthField(oLen) + 1;

        byte[] bytes = new byte[len];
        bytes[0] = 0x10 | 0x20; // ASN sequence & constructed
        int offset = writeLengthField(bytes, oLen);

        if (len - oLen != offset) {
            return null;
        }

        for (byte[] b : byteArrays) {
            if (b == null) {
                break;
            }

            System.arraycopy(b, 0, bytes, offset, b.length);
            offset += b.length;
        }

        return bytes;
    }

    private static int writeLengthField(byte[] bytes, int len) {
        if (len < 127) {
            bytes[1] = (byte) len;
            return 2;
        }

        int lenOfLenField = getLengthOfLengthField(len);
        bytes[1] = (byte) ((lenOfLenField - 1) | 0x80); // record length of the length field

        for (int i = lenOfLenField; i >= 2; i--) { // write the length
            bytes[i] = (byte) (len >> ((lenOfLenField - i) * 8));
        }

        return lenOfLenField + 1;
    }

    private static int getLengthOfLengthField(int len) {
        if (len <= 127) { // highest bit is zero, one byte is enough
            return 1;
        } else if (len <= 0xFF) { // highest bit is 1, two bytes in the form {0x81, 0xab}
            return 2;
        } else if (len <= 0xFFFF) { // three bytes in the form {0x82, 0xab, 0xcd}
            return 3;
        } else if (len <= 0xFFFFFF) { // four bytes in the form {0x83, 0xab, 0xcd, 0xef}
            return 4;
        } else { // five bytes in the form {0x84, 0xab, 0xcd, 0xef, 0xgh}
            return 5;
        }
    }

    private static int getOIDCompLength(int comp) {
        if (comp <= 0x7F) {
            return 1;
        } else if (comp <= 0x3FFF) {
            return 2;
        } else if (comp <= 0x1FFFFF) {
            return 3;
        } else if (comp <= 0xFFFFFFF) {
            return 4;
        } else {
            return 5;
        }
    }

    private static int writeOIDComp(int comp, byte[] bytes, int offset) {
        int len = getOIDCompLength(comp);
        int off = offset;
        for (int i = len - 1; i > 0; i--) {
            bytes[off++] = (byte) ((comp >>> i * 7) | 0x80);
        }

        bytes[off++] = (byte) (comp & 0x7F);

        return off;
    }
}
