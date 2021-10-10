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
package ai.djl.util;

import ai.djl.ndarray.BytesSupplier;
import java.io.Serializable;

/**
 * A class implements {@code JsonSerializable} indicates it can be serialized into a json string.
 */
public interface JsonSerializable extends Serializable, BytesSupplier {

    /**
     * Returns a json presentation of the object.
     *
     * @return a json string
     */
    String toJson();
}
