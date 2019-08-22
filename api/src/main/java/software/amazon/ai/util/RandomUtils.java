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
package software.amazon.ai.util;

import java.util.Random;

/** A class hold a static instance of {@link Random} object. */
public final class RandomUtils {

    public static final Random RANDOM = new Random();

    private RandomUtils() {}

    public static double random() {
        return RANDOM.nextDouble();
    }

    public static double nextGaussian() {
        return RANDOM.nextGaussian();
    }

    public static int nextInt() {
        return RANDOM.nextInt();
    }

    public static int nextInt(int bound) {
        return RANDOM.nextInt(bound);
    }
}
