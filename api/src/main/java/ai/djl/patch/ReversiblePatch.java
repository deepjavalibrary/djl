/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.patch;

/** A {@link Patch} that can be reversed. */
public abstract class ReversiblePatch extends Patch {

    /**
     * Returns a new {@link Patch} that reverses the effect of this one.
     *
     * <p>For a {@link ParamPatch}, it is equivalent to scaling by -1.
     *
     * @return a new {@link Patch} that reverses the effect of this one.
     */
    public abstract ParamPatch reverse();
}
