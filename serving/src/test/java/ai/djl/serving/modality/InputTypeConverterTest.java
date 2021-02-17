/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.serving.modality;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;

import ai.djl.modality.Input;
import ai.djl.modality.cv.Image;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.util.PairList;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.testng.annotations.Test;

public class InputTypeConverterTest {
    @Test
    public void testInputToImageConversion()
            throws ConversionException, IOException, URISyntaxException {

        InputTypeConverter converter = new InputTypeConverter();

        ModelInfo modelInfo =
                new ModelInfo("testModel", Image.class, Object.class, null, null, 0, 0, 0, 0);
        Input input = new Input("unittest-testInputToImageConversion");
        Path path = Paths.get(ClassLoader.getSystemResource("0.png").toURI());
        byte[] raw = java.nio.file.Files.readAllBytes(path);
        PairList<String, byte[]> pair = new PairList<>();
        pair.add("data", raw);
        input.setContent(pair);
        Image img = (Image) converter.convertToInputData(modelInfo, input);
        assertNotNull(img);
    }

    @Test
    public void testNoConversionWhenInputTypeIsNull()
            throws ConversionException, IOException, URISyntaxException {

        InputTypeConverter converter = new InputTypeConverter();

        ModelInfo modelInfo =
                new ModelInfo("testModel", null, Object.class, null, null, 0, 0, 0, 0);
        Input input = new Input("unittest-testInputToImageConversion");

        Input out = (Input) converter.convertToInputData(modelInfo, input);
        assertNotNull(out);
        assertEquals(out, input);
    }
}
