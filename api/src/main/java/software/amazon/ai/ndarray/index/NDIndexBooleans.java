package software.amazon.ai.ndarray.index;

import software.amazon.ai.ndarray.NDArray;

public class NDIndexBooleans implements NDIndexElement {

    private NDArray index;

    public NDIndexBooleans(NDArray index) {
        this.index = index;
    }

    public NDArray getIndex() {
        return index;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return index.getShape().dimension();
    }
}
