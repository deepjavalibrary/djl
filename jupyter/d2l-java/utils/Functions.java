import java.util.function.Function;

public class Functions {
    // Applies the function `func` to `x` element-wise
    // Returns a new float array with the result
    public static float[] callFunc(float[] x, Function<Float, Float> func) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = func.apply(x[i]);
        }
        return y;
    }
    
    // ScatterTrace.builder() does not support float[],
    // so we must convert to a double array first
    public static double[] floatToDoubleArray(float[] x) {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            ret[i] = x[i];
        }
        return ret;
    }
}