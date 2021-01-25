package ai.djl.serving.wlm;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ModelInfoTest {

    @Test
    public void testQueueSizeIsSet() {
	ModelInfo modelInfo=new ModelInfo("", "", null, 4711);
	Assert.assertEquals(4711, modelInfo.getQueueSize());
    }
    
}
