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
package ai.djl.arrayfire.jna;

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

/** A class containing utilities to interact with the ArrayFire Engine's JNA layer. */
@SuppressWarnings({"missingjavadocmethod", "MethodName"})
public interface ArrayFireLibrary extends Library {

    // URL: https://arrayfire.org/docs/group__c__api__mat.htm
    // AF Array functions array.h
    int af_create_array(PointerByReference handle, Pointer data, int ndims, long[] dims, int dtype);

    int af_get_type(IntBuffer buf, Pointer handle);

    int af_get_data_ref_count(IntBuffer num, Pointer handle);

    int af_eval(Pointer handle);

    int af_get_data_ptr(Pointer data, Pointer handle);

    int af_get_numdims(IntBuffer buf, Pointer handle);

    int af_get_dims(LongBuffer d0, LongBuffer d1, LongBuffer d2, LongBuffer d3, Pointer handle);

    int af_release_array(Pointer handle);

    // AF basic device.h
    int af_info();

    int af_set_backend(int backend);

    int af_set_device(int device);

    // math ops arith.h
    int af_add(Pointer out, Pointer lhs, Pointer rhs, boolean batch);
}
