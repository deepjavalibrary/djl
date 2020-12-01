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

package ai.djl.paddlepaddle.jna;

import com.sun.jna.Library;
import com.sun.jna.Pointer;

/** The Paddle JNA library. */
@SuppressWarnings({"missingjavadocmethod", "methodname"})
public interface PaddleLibrary extends Library {

    // Paddle Inference config
    Pointer PD_NewAnalysisConfig();

    // config modification
    void PD_DeleteAnalysisConfig(Pointer config);

    void PD_SetCpuMathLibraryNumThreads(Pointer config, int cpu_math_library_num_threads);

    void PD_SwitchUseFeedFetchOps(Pointer config, boolean x);

    void PD_SwitchSpecifyInputNames(Pointer config, boolean x);

    void PD_SwitchIrDebug(Pointer config, boolean x);

    void PD_EnableMKLDNN(Pointer config);

    void PD_EnableMkldnnBfloat16(Pointer config);

    void PD_EnableMkldnnQuantizer(Pointer config);

    void PD_SetMkldnnCacheCapacity(Pointer config, int capacity);

    // config value check
    boolean PD_UseFeedFetchOpsEnabled(Pointer config);

    // Paddle Tensor
    Pointer PD_NewPaddleTensor();

    void PD_DeletePaddleTensor(Pointer tensor);

    String PD_GetPaddleTensorName(Pointer tensor);

    void PD_SetPaddleTensorName(Pointer tensor, String name);

    void PD_SetPaddleTensorDType(Pointer tensor, int dtype);

    void PD_SetPaddleTensorData(Pointer tensor, Pointer buf);

    void PD_SetPaddleTensorShape(Pointer tensor, int[] shape, int size);

    int PD_GetPaddleTensorDType(Pointer tensor);

    Pointer PD_GetPaddleTensorData(Pointer tensor);

    int[] PD_GetPaddleTensorShape(Pointer tensor, int[] shape);

    Pointer PD_NewPaddleBuf();

    void PD_PaddleBufReset(Pointer buf, Pointer data, int size);
}
