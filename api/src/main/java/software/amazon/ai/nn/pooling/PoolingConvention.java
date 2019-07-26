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
package software.amazon.ai.nn.pooling;

public enum PoolingConvention {
    VALID,
    FULL;

    public long getOutput(long input, long kernel, long stride, long pad) {
        double output = input + 2 * pad - kernel / stride;
        switch (this) {
            case VALID:
                return (long) Math.floor(output) + 1;
            case FULL:
                return (long) Math.ceil(output) + 1;
            default:
                throw new IllegalStateException();
        }
    }
}
