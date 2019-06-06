package software.amazon.ai.ndarray.index;

/** An NDIndexElement to return all values in a particular dimension. */
public class NDIndexAll implements NDIndexElement {

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return 1;
    }
}
