package ai.djl.timeseries.distribution;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.testing.Assertions;
import org.testng.annotations.Test;

public class DistributionTest {

    @Test
    public void testNegativeBinomial() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray mu = manager.create(new float[]{1000f, 1f});
            NDArray alpha = manager.create(new float[]{1f, 2f});
            mu.setName("mu");
            alpha.setName("alpha");
            Distribution negativeBinomial = NegativeBinomial
                    .builder()
                    .setDistrArgs(new NDList(mu, alpha))
                    .build();

            NDArray expected = manager.create(new float[]{-6.9098f, -1.6479f});
            NDArray real = negativeBinomial.logProb(manager.create(new float[]{1f, 1f}));
            Assertions.assertAlmostEquals(real, expected);
        }
    }

    @Test
    public void testStudentT() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray mu = manager.create(new float[]{1000f, -1000f});
            NDArray sigma = manager.create(new float[]{1f, 2f});
            NDArray nu = manager.create(new float[]{4.2f, 3f});
            mu.setName("mu");
            sigma.setName("sigma");
            nu.setName("nu");
            Distribution studentT = StudentT
                    .builder()
                    .setDistrArgs(new NDList(mu, sigma, nu))
                    .build();

            NDArray expected = manager.create(new float[]{-0.9779f, -1.6940f});
            NDArray real = studentT.logProb(manager.create(new float[]{1000f, -1000f}));
            Assertions.assertAlmostEquals(real, expected);
        }
    }
}
