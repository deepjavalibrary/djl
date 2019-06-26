package org.apache.mxnet.nn.core;

import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;
import software.amazon.ai.Initializer;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDFactory;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.core.Linear;

public class MxLinear extends MxNNBlock implements Linear {

    private NDArray weight;
    private NDArray bias;

    private int units;
    private int inUnits;

    public MxLinear(int units, int inUnits) {
        this.opName = "FullyConnected";
        this.units = units;
        this.inUnits = inUnits;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getInputShape() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public List<NDArray> getDirectParameters() {
        return Arrays.asList(weight, bias);
    }

    /** {@inheritDoc} */
    @Override
    public void initialize(NDFactory factory, Initializer initializer) {
        weight = factory.create(new DataDesc(new Shape(units, inUnits)));
        bias = factory.create(new DataDesc(new Shape(units)));
        initializer.initialize(getDirectParameters());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray forward(NDArray data) {
        NDList inputs = new NDList(data, weight, bias);
        MxOpParams params = new MxOpParams();
        params.add("num_hidden", "1");
        return forward(inputs, params).get(0);
    }
}
