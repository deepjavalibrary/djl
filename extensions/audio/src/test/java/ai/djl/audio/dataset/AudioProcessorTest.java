package ai.djl.audio.dataset;

import ai.djl.Device;
import ai.djl.audio.AudioUtils;
import ai.djl.audio.featurizer.AudioNormalizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

public class AudioProcessorTest {

    static String filePath = "src/test/resources/61-70968-0000.flac";

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
}
