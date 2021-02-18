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
package ai.djl.serving.loading;

import ai.djl.modality.Input;
import ai.djl.modality.Output;

/**
 * creates a criteria object to lookup for model using a modelURL String.
 *
 * @author erik.bamberg@web.de
 */
public class URLOnlyModelCriteriaParser extends ModelCriteriaParser<String> {

    /** {@inheritDoc}} */
    @Override
    protected ModelCriteriaParser<String>.Parameters parseInput(String input) {
        Parameters param = new Parameters();
        param.setModelUrl(input);
        param.setInputType(Input.class);
        param.setOutputType(Output.class);
        return param;
    }
}
