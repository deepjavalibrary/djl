package ai.djl.basicdataset.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import org.bytedeco.javacv.*;

import java.nio.Buffer;

public class AudioData {



    public static float[] toFloat(String path) throws FrameGrabber.Exception {
        try(FFmpegFrameGrabber audioGrabber = new FFmpegFrameGrabber(path)){
            audioGrabber.start();
            Frame frame;
            while((frame = audioGrabber.grabFrame()) != null){
                Buffer[] buffers = frame.samples;

            }
        }
        catch (FrameGrabber.Exception e){
            e.printStackTrace();
        }

    }

}
