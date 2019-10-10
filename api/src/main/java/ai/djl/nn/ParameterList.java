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
package ai.djl.nn;

import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.util.List;
import java.util.Map;

/** Represents a set of names and Parameters. */
public class ParameterList extends PairList<String, Parameter> {

    public ParameterList() {}

    public ParameterList(int initialCapacity) {
        super(initialCapacity);
    }

    public ParameterList(List<String> keys, List<Parameter> values) {
        super(keys, values);
    }

    public ParameterList(List<Pair<String, Parameter>> list) {
        super(list);
    }

    public ParameterList(Map<String, Parameter> map) {
        super(map);
    }
}
