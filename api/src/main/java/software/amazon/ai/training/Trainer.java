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
package software.amazon.ai.training;

import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.training.optimizer.Optimizer;

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
