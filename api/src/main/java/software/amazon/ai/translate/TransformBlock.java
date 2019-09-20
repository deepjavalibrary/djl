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

import java.util.Arrays;
import java.util.List;
import software.amazon.ai.ndarray.NDList;

public class TransformBlock {
    List<Transform> transforms;

    public TransformBlock(Transform... transforms) {
        this.transforms = Arrays.asList(transforms);
    }

    public TransformBlock(List<Transform> transforms) {
        this.transforms = transforms;
    }

    public NDList transform(NDList input, boolean close) {
        if (input.size() != transforms.size()) {
            throw new IllegalArgumentException(
                    "Input NDList size "
                            + input.size()
                            + " mismatch number of Transform "
                            + transforms.size());
        }
        NDList result = new NDList(input.size());
        for (int i = 0; i < input.size(); i++) {
            result.add(transforms.get(i).transform(input.get(i), close));
            // close intermediate NDArray
            input.close();
        }
        return result;
    }
}
