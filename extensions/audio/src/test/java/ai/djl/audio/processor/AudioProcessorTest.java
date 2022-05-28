package ai.djl.audio.processor;

import ai.djl.Device;
import ai.djl.audio.AudioUtils;
import ai.djl.audio.dataset.AudioData;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

public class AudioProcessorTest {

    private static String filePath = "src/test/resources/test.wav";
    private static float eps = 1e-3f;

    @Test
    public static void testAudioNormalizer() {
        NDManager manager = NDManager.newBaseManager(Device.cpu());
        AudioData.Configuration configuration =
                new AudioData.Configuration()
                        .setProcessorList(Arrays.asList(new AudioNormalizer(-20)));
        AudioData testData = new AudioData(configuration);
        NDArray samples = testData.getPreprocessedData(manager, filePath);
        Assert.assertEquals(AudioUtils.rmsDb(samples), -20.0f, 1e-3);
    }

    @Test
    public static void testLinearSpecgram() {
        NDManager manager = NDManager.newBaseManager(Device.cpu());
        AudioData.Configuration configuration =
                new AudioData.Configuration()
                        .setSampleRate(16000)
                        .setProcessorList(
                                Arrays.asList(
                                        new AudioNormalizer(-20),
                                        new LinearSpecgram(10, 20, 16000)));
        AudioData testData = new AudioData(configuration);
        NDArray samples = testData.getPreprocessedData(manager, filePath);
        Assert.assertTrue(samples.getShape().equals(new Shape(161, 838)));
        Assert.assertEquals(samples.get("0,0").toFloatArray()[0], -15.4571f, eps);
    }
}
