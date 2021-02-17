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
package ai.djl.serving.modality;

/**
 * Exception is throw when a type-conversion fails.
 *
 * @author erik.bamberg@web.de
 */
public class ConversionException extends Exception {

    /** the serialVersionUID. */
    private static final long serialVersionUID = -4408856133870325345L;

    /** default constructor. */
    public ConversionException() {}

    /**
     * construct with a Message.
     *
     * @param message error cause.
     */
    public ConversionException(String message) {
        super(message);
        // TODO Auto-generated constructor stub
    }

    /**
     * construct with a cause.
     *
     * @param cause error cause.
     */
    public ConversionException(Throwable cause) {
        super(cause);
    }

    /**
     * construct with a Message and cause.
     *
     * @param message error cause.
     * @param cause error cause.
     */
    public ConversionException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * full construct with a Message and cause.
     *
     * @param message error cause.
     * @param cause error cause.
     * @param enableSuppression error cause.
     * @param writableStackTrace error cause.
     */
    public ConversionException(
            String message,
            Throwable cause,
            boolean enableSuppression,
            boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
