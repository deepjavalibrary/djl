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
package software.amazon.ai.translate;

import java.util.ArrayList;
import java.util.List;
import software.amazon.ai.ndarray.NDArray;

public class Pipeline implements Transform {

    private List<Transform> transforms;

    public Pipeline() {
        transforms = new ArrayList<>();
    }

    public Pipeline add(Transform transform) {
        transforms.add(transform);
        return this;
    }

    @Override
    public NDArray transform(NDArray array, boolean close) {
        if (transforms.isEmpty()) {
            return array;
        }

        NDArray ret = transforms.get(0).transform(array, close);
        for (int i = 1; i < transforms.size(); ++i) {
            ret = transforms.get(i).transform(ret, true);
        }
        return ret;
    }
}
