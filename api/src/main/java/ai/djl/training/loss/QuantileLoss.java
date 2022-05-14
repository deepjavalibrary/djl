package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;

/**
 * {@code QuantileLoss} calculates the Weighted Quantile Loss between labels and predictions.
 * It is useful for regression problems where you wish to estimate a particular quantile.
 * For example, to target the P90 instantiate {@code new QuantileLoss("P90",0.90)},
 * but you'd better have plenty of data and/or an easy problem if you want to get near 0% or 100%.
 *
 * <pre>
 * def quantile_loss(target, forecast, q):
 *     return (2*np.sum(np.abs((forecast-target)*((target<=forecast)-q))))
 *
 * Reference: https://bibinmjose.github.io/2021/03/08/errorblog.html
 */
public class QuantileLoss extends Loss {

    private Number quantile;
    private boolean weighted;

    public QuantileLoss(String name, double quantile, boolean weighted) {
        super(name);
        this.quantile = quantile;
        this.weighted = weighted;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray pred = predictions.singletonOrThrow();
        NDArray target = labels.singletonOrThrow().reshape(pred.getShape());
        NDArray loss = pred.sub(target).mul(target.lte(pred).toType(DataType.FLOAT32, false).sub(quantile)).abs().mul(2);
        if (weighted) {
            loss = loss.div(target.sum());
        } else {
            loss = loss.div(target.size());
        }
        return loss;
    }
}
