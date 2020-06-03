// Saved in the d2l-java package for later use
class StopWatch {

    // Record multiple running times.
    private ArrayList<Double> times;
    private long tik;

    public StopWatch() {
        times = new ArrayList();
        start();
    }

    public void start() {
        tik = System.nanoTime();
    }

    public double stop() {
        times.add(nanoToSec(System.nanoTime() - tik));
        return times.get(times.size() - 1);
    }

    // Return average time
    public double avg() {
        return sum() / times.size();
    }

    // Return the sum of time
    public double sum() {
        double sum = 0;
        for (double d : times) {
            sum += d;
        }
        return sum;
    }

    // Return the accumulated times
    public ArrayList cumsum() {
        ArrayList<Double> cumsumList = new ArrayList();
        double currentSum = 0;
        for (Double d : times) {
            currentSum += d;
            cumsumList.add(currentSum);
        }
        return times;
    }

    // Convert nano seconds to seconds
    private double nanoToSec(long nanosec) {
        return (double) nanosec / 1E9;
    }
}

