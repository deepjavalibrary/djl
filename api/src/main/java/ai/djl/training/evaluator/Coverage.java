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

    public Coverage() {
        this("Coverage", 1);
    }

    public Coverage(String name, int axis) {
        super(name, axis);
    }

    @Override
    protected Pair<Long, NDArray> accuracyHelper(NDList labels, NDList predictions) {
        NDArray labl = labels.head();
        NDArray pred = predictions.head();
        return new Pair<>(labl.size(), labl.lt(pred));
    }
}
