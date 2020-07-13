/**
 * Sums a list of numbers over time.
 *
 * Defined in Ch 3.6 Softmax Reg. from Scratch
 */
public class Accumulator {
    float[] data;

    /**
     * Constructor for Accumulator.
     *
     * @param n is the size of the array
     */
    public Accumulator(int n) {
        data = new float[n];
    }

    /* Adds a set of numbers to the array */
    public void add(float[] args) {
        for (int i = 0; i < args.length; i++) {
            data[i] += args[i];
        }
    }

    /* Resets the array */
    public void reset() {
        Arrays.fill(data, 0f);
    }

    /* Returns the data point at the given index */
    public float get(int index) {
        return data[index];
    }
}