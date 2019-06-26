package org.apache.mxnet.nn;

import org.apache.mxnet.nn.core.MxLinear;
import software.amazon.ai.nn.NNIndex;
import software.amazon.ai.nn.core.Linear;

public class MxNNIndex extends NNIndex {

    /** {@inheritDoc} */
    @Override
    public Linear linear(int units, int inUnits) {
        return new MxLinear(units, inUnits);
    }
}
