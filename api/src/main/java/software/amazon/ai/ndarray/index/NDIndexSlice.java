package software.amazon.ai.ndarray.index;

/** An NDIndexElement that returns a range of values in the specified dimension. */
public class NDIndexSlice implements NDIndexElement {

    private Integer min;

    private Integer max;

    private Integer step;

    public NDIndexSlice(Integer min, Integer max, Integer step) {
        this.min = min;
        this.max = max;
        this.step = step;
    }

    public Integer getMin() {
        return min;
    }

    public Integer getMax() {
        return max;
    }

    public Integer getStep() {
        return step;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return 1;
    }
}
