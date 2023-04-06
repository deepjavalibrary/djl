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
package ai.djl.modality;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

/** A class stores the generic inference results. */
public class Output extends Input {

    private static final long serialVersionUID = 1L;

    private int code;
    private String message;

    /** Constructs a {@code Output} instance. */
    public Output() {
        this(200, "OK");
    }

    /**
     * Constructs a {@code Output} with specified {@code requestId}, {@code code} and {@code
     * message}.
     *
     * @param code the status code of the output
     * @param message the status message of the output
     */
    public Output(int code, String message) {
        this.code = code;
        this.message = message;
    }

    /**
     * Returns the status code of the output.
     *
     * @return the status code of the output
     */
    public int getCode() {
        return code;
    }

    /**
     * Sets the status code of the output.
     *
     * @param code the status code of the output
     */
    public void setCode(int code) {
        this.code = code;
    }

    /**
     * Returns the status code of the output.
     *
     * @return the status code of the output
     */
    public String getMessage() {
        return message;
    }

    /**
     * Sets the status message of the output.
     *
     * @param message the status message of the output
     */
    public void setMessage(String message) {
        this.message = message;
    }

    /**
     * Encodes all data in the output to a binary form.
     *
     * @return the binary encoding
     * @throws IOException if it fails to encode part of the data
     */
    @Override
    public byte[] encode() throws IOException {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            DataOutputStream os = new DataOutputStream(baos);
            os.writeLong(serialVersionUID);
            encodeInputBase(os);

            os.writeInt(code);
            os.writeUTF(message);

            return baos.toByteArray();
        }
    }

    /**
     * Decodes the output from {@link #encode()}.
     *
     * @param is the data to decode from
     * @return the decoded output
     * @throws IOException if it fails to decode part of the output
     */
    public static Output decode(InputStream is) throws IOException {
        try (DataInputStream dis = new DataInputStream(is)) {
            if (serialVersionUID != dis.readLong()) {
                throw new IllegalArgumentException("Invalid Input version");
            }

            Output output = new Output();
            decodeInputBase(dis, output);

            output.code = dis.readInt();
            output.message = dis.readUTF();

            return output;
        }
    }

    /**
     * Checks for deep equality with another output.
     *
     * @param o the other output.
     * @return whether they and all properties, content, and data are equal
     */
    @Override
    public boolean deepEquals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        if (!super.deepEquals(o)) {
            return false;
        }
        Output output = (Output) o;
        return code == output.code && Objects.equals(message, output.message);
    }
}
