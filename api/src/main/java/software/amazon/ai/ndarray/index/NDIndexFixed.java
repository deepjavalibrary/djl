package software.amazon.ai.ndarray.index;

/** An NDIndexElement that returns only a specific value in the corresponding dimension. */
public class NDIndexFixed implements NDIndexElement {

    private int index;

    public NDIndexFixed(int index) {
        this.index = index;
    }

    public int getIndex() {
        return index;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return 1;
    }
}
