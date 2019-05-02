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
package com.amazon.ai.training;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.engine.Engine;

public interface Trainer {

    static Trainer newInstance(Model model) {
        return newInstance(model, Context.defaultContext());
    }

    static Trainer newInstance(Model model, Context context) {
        return Engine.getInstance().newTrainer(model, context);
    }

    Estimator getEstimator();

    void setEstimator(Estimator estimator);

    Optimizer getOptimizer();

    void setOptimizer(Optimizer optimizer);

    ModelSaver getModelSaver();

    void setModelSaver(ModelSaver modelSaver);

    void checkpoint();
}
