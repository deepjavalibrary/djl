package com.amazon.ai.nn;

import com.amazon.ai.nn.core.Linear;

public abstract class NNIndex {

    public abstract Linear linear(int units, int inUnits);
}
