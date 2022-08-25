/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.training.evaluator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;

/**
 * Coverage for a Regression problem: it measures the percent of predictions greater than the actual
 * target, to determine whether the predictor is over-forecasting or under-forecasting. e.g. 0.50 if
 * we predict near the median of the distribution.
 *
 * <pre>
 *  def coverage(target, forecast):
 *     return (np.mean((target &lt; forecast)))
 * </pre>
 *
 * <a href="https://bibinmjose.github.io/2021/03/08/errorblog.html">...</a>
 */
public class Coverage extends AbstractAccuracy {

    /**
     * Creates an evaluator that measures the percent of predictions greater than the actual target.
     */
    public Coverage() {
        this("Coverage", 1);
    }

    /**
     * Creates an evaluator that measures the percent of predictions greater than the actual target.
     *
     * @param name the name of the evaluator, default is "Coverage"
     * @param axis the axis along which to count the correct prediction, default is 1
     */
    public Coverage(String name, int axis) {
        super(name, axis);
    }

    /** {@inheritDoc} */
    @Override
    protected Pair<Long, NDArray> accuracyHelper(NDList labels, NDList predictions) {
        NDArray labl = labels.head();
        NDArray pred = predictions.head();
        return new Pair<>(labl.size(), labl.lt(pred));
    }
}
