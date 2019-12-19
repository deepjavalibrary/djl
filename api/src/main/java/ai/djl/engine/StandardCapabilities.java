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
package ai.djl.engine;

/** Constant definitions for the standard capability. */
public final class StandardCapabilities {

    public static final String CUDA = "CUDA";
    public static final String CUDNN = "CUDNN";
    public static final String MKL = "MKL";
    public static final String MKLDNN = "MKLDNN";
    public static final String OPENMP = "OPENMP";

    private StandardCapabilities() {}
}
