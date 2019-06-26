package software.amazon.ai.nn;

import software.amazon.ai.nn.core.Linear;

public abstract class NNIndex {

    public abstract Linear linear(int units, int inUnits);
}
