package org.apache.mxnet.engine;
// CHECKSTYLE:OFF:AvoidStaticImport

import static org.powermock.api.mockito.PowerMockito.mockStatic;

import com.amazon.ai.Context;
import com.amazon.ai.test.MockMxnetLibrary;
import java.io.File;
import java.io.IOException;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.apache.commons.io.FileUtils;
import org.apache.mxnet.jna.LibUtils;
import org.apache.mxnet.jna.MxnetLibrary;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

// CHECKSTYLE:ON:AvoidStaticImport

@PrepareForTest(LibUtils.class)
public class MxEngineTest extends PowerMockTestCase {

    @BeforeClass
    public void prepare() throws IOException {
        mockStatic(LibUtils.class);
        MxnetLibrary library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
        FileUtils.deleteQuietly(new File("build/tmp/A-0122.params"));
        FileUtils.deleteQuietly(new File("build/tmp/A-0001.params"));
    }

    @AfterClass
    public void postProcessing() throws IOException {
        FileUtils.deleteQuietly(new File("build/tmp/A-0122.params"));
        FileUtils.deleteQuietly(new File("build/tmp/A-0001.params"));
    }

    @Test
    public void testGetGpuMemory() {
        MxEngine engine = new MxEngine();
        MemoryUsage usage = engine.getGpuMemory(Context.gpu(0));
        Assert.assertEquals(usage.getUsed(), 100);
        Assert.assertEquals(usage.getMax(), 1000);
    }

    @Test
    public void testDefaultContext() {
        MxEngine engine = new MxEngine();
        Assert.assertEquals(engine.defaultContext(), Context.gpu());
    }

    @Test
    public void testGetVersion() {
        MxEngine engine = new MxEngine();
        Assert.assertEquals(engine.getVersion(), "1.5.0");
    }

    @Test
    public void testLoadModel() throws IOException {
        MxEngine engine = new MxEngine();
        String fileLocation = "build/tmp/";
        String modelName = "A";
        Files.createFile(Paths.get(fileLocation + modelName + "-0122.params"));
        String result = loadModel(engine, fileLocation, modelName, -1);
        FileUtils.forceDelete(new File(fileLocation + modelName + "-0122.params"));
        Assert.assertEquals(result, "A-0122.params");
        Files.createFile(Paths.get(fileLocation + modelName + "-0001.params"));
        result = loadModel(engine, fileLocation, modelName, -1);
        FileUtils.forceDelete(new File(fileLocation + modelName + "-0001.params"));
        Assert.assertEquals(result, "A-0001.params");
        result = loadModel(engine, fileLocation, modelName, 122);
        Assert.assertEquals(result, "A-0122.params");
        Files.createFile(Paths.get(fileLocation + modelName + "-0122.params"));
        Files.createFile(Paths.get(fileLocation + modelName + "-0001.params"));
        result = loadModel(engine, fileLocation, modelName, -1);
        Assert.assertEquals(result, "A-0122.params");
    }

    private String loadModel(MxEngine engine, String location, String modelName, int epoch)
            throws IOException {
        MxModel model = (MxModel) engine.loadModel(new File(location), modelName, epoch);
        // In JNA.MXNDArrayLoad function, file name is stored as the first param name in Model
        String paramPath = model.getParameters().get(0).getKey();
        return new File(paramPath).getName();
    }
}
