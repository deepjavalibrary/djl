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

package org.tensorflow;

import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import org.tensorflow.engine.TfNDFactory;

public final class HelloWorld {

    private HelloWorld() {}

    @SuppressWarnings("PMD.SystemPrintln")
    public static void main(String[] args) {
        try (NDFactory factory = TfNDFactory.SYSTEM_FACTORY.newSubFactory()) {
            NDArray a = factory.create(new float[] {1.0f, 2.0f});
            System.out.println(a.getShape());
        }
    }
}
