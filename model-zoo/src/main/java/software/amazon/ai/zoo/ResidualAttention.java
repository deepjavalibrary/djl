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
package software.amazon.ai.zoo;

import software.amazon.ai.ndarray.NDList;

public class ResidualAttention {

    boolean isPretrained() {
        return false;
    }

    public enum InputSize {
        INPUT_SIZE_32,
        INPUT_SIZE_224
    }

    public static final class ResidualAttentionSpec implements NetworkSpec {

        @Override
        public boolean loadPretrained() {
            return false;
        }

        @Override
        public int getNumOfLayers() {
            return 0;
        }

        @Override
        public int[] getLayers() {
            return new int[0];
        }

        @Override
        public int[] getSupportedLayers() {
            return new int[0];
        }

        @Override
        public int[] getChannels() {
            return new int[0];
        }

        @Override
        public int getVersion() {
            return 0;
        }

        @Override
        public String getBlockVersion() {
            return null;
        }

        @Override
        public NDList normalize(NDList list) {
            return null;
        }

        @Override
        public NDList transform(NDList list) {
            return null;
        }
    }
}
