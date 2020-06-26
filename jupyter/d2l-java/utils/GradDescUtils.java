import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.util.ArrayList;
import java.util.function.BiFunction;
import java.util.function.Function;

public class GradDescUtils {
    public static void plotGD(float[] x, float[] y, float[] segment, Function<Float, Float> func,
                              int width, int height) {
        // Function Line
        ScatterTrace trace = ScatterTrace.builder(Functions.floatToDoubleArray(x),
                Functions.floatToDoubleArray(y))
                .mode(ScatterTrace.Mode.LINE)
                .build();

        // GD Line
        ScatterTrace trace2 = ScatterTrace.builder(Functions.floatToDoubleArray(segment),
                Functions.floatToDoubleArray(Functions.callFunc(segment, func)))
                .mode(ScatterTrace.Mode.LINE)
                .build();

        // GD Points
        ScatterTrace trace3 = ScatterTrace.builder(Functions.floatToDoubleArray(segment),
                Functions.floatToDoubleArray(Functions.callFunc(segment, func)))
                .build();

        Layout layout = Layout.builder()
                .height(height)
                .width(width)
                .showLegend(false)
                .build();

        display(new Figure(layout, trace, trace2, trace3));
    }

    public static void showTrace(float[] res, Function<Float, Float> f, NDManager manager) {
        float n = 0;
        for (int i = 0; i < res.length; i++) {
            if (Math.abs(res[i]) > n) {
                n = Math.abs(res[i]);
            }
        }
        NDArray fLineND = manager.arange(-n, n, 0.01f);
        float[] fLine = fLineND.toFloatArray();
        plotGD(fLine, Functions.callFunc(fLine, f), res, f, 500, 400);
    }

    public static class Weights {
        public float x1, x2;

        public Weights(float x1, float x2) {
            this.x1 = x1;
            this.x2 = x2;
        }
    }

    /* Optimize a 2D objective function with a customized trainer. */
    public static ArrayList<Weights> train2d(Function<Float[], Float[]> trainer, int steps) {
        // s1 and s2 are internal state variables and will
        // be used later in the chapter
        float x1 = -5f, x2 = -2f, s1 = 0f, s2 = 0f;
        ArrayList<Weights> results = new ArrayList<>();
        results.add(new Weights(x1, x2));
        for (int i = 1; i < steps + 1; i++) {
            Float[] step = trainer.apply(new Float[]{x1, x2, s1, s2});
            x1 = step[0];
            x2 = step[1];
            s1 = step[2];
            s2 = step[3];
            results.add(new Weights(x1, x2));
            System.out.printf("epoch %d, x1 %f, x2 %f\n", i, x1, x2);
        }
        return results;
    }


    /* Show the trace of 2D variables during optimization. */
    public static void showTrace2d(BiFunction<Float, Float, Float> f, ArrayList<Weights> results) {
        // TODO: add when tablesaw adds support for contour and meshgrids
    }

    public static Figure plotGammas(float[] time, float[] gammas,
                                  int width, int height) {
        double[] gamma1 = new double[time.length];
        double[] gamma2 = new double[time.length];
        double[] gamma3 = new double[time.length];
        double[] gamma4 = new double[time.length];

        // Calculate all gammas over time
        for (int i = 0; i < time.length; i++) {
            gamma1[i] = Math.pow(gammas[0], i);
            gamma2[i] = Math.pow(gammas[1], i);
            gamma3[i] = Math.pow(gammas[2], i);
            gamma4[i] = Math.pow(gammas[3], i);
        }

        // Gamma 1 Line
        ScatterTrace gamma1trace = ScatterTrace.builder(Functions.floatToDoubleArray(time),
                gamma1)
                .mode(ScatterTrace.Mode.LINE)
                .name(String.format("gamma = %.2f", gammas[0]))
                .build();

        // Gamma 2 Line
        ScatterTrace gamma2trace = ScatterTrace.builder(Functions.floatToDoubleArray(time),
                gamma2)
                .mode(ScatterTrace.Mode.LINE)
                .name(String.format("gamma = %.2f", gammas[1]))
                .build();

        // Gamma 3 Line
        ScatterTrace gamma3trace = ScatterTrace.builder(Functions.floatToDoubleArray(time),
                gamma3)
                .mode(ScatterTrace.Mode.LINE)
                .name(String.format("gamma = %.2f", gammas[2]))
                .build();

        // Gamma 4 Line
        ScatterTrace gamma4trace = ScatterTrace.builder(Functions.floatToDoubleArray(time),
                gamma4)
                .mode(ScatterTrace.Mode.LINE)
                .name(String.format("gamma = %.2f", gammas[3]))
                .build();

        Axis xAxis = Axis.builder()
                .title("time")
                .build();

        Layout layout = Layout.builder()
                .height(height)
                .width(width)
                .xAxis(xAxis)
                .build();

        return new Figure(layout, gamma1trace, gamma2trace, gamma3trace, gamma4trace);
    }
}