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
package ai.djl.modality.rl;

import ai.djl.ndarray.NDList;
import ai.djl.util.RandomUtils;
import java.util.ArrayList;

/** Contains the available actions that can be taken in an {@link ai.djl.modality.rl.env.RlEnv}. */
public class ActionSpace extends ArrayList<NDList> {

    private static final long serialVersionUID = 8683452581122892189L;

    /**
     * Returns a random action.
     *
     * @return a random action
     */
    public NDList randomAction() {
        return get(RandomUtils.nextInt(size()));
    }
}
