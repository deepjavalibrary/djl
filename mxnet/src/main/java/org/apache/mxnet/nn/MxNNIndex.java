package org.apache.mxnet.nn;

import com.amazon.ai.nn.NNIndex;
import com.amazon.ai.nn.core.Linear;
import org.apache.mxnet.nn.core.MxLinear;

public class MxNNIndex extends NNIndex {

    /** {@inheritDoc} */
    @Override
    public Linear linear(int units, int inUnits) {
        return new MxLinear(units, inUnits);
    }
}
