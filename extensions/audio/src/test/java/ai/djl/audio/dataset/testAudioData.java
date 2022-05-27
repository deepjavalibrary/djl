package ai.djl.audio.dataset;

import org.bytedeco.javacv.FrameGrabber;
import org.testng.annotations.Test;

import java.util.List;

public class testAudioData {
    
    @Test
    public void testAudioDatatofloat() throws FrameGrabber.Exception {
        String path = "src/test/resources/61-70968-0000.flac";
        AudioData.Configuration configuration = new AudioData.Configuration();
        configuration.setPath(path).setSampleRate(40000);
        AudioData audioData = new AudioData(configuration);
        List<Float> floatList = audioData.getFloatList();
        for (float f : floatList) {
            System.out.println(f);
        }
        System.out.println(floatList.size());
    }
}
