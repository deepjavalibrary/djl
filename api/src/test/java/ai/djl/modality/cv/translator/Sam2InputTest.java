/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.translator;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Sam2InputTest {

    @Test
    public void test() throws IOException {
        Path file = Paths.get("../examples/src/test/resources/kitten.jpg");
        Image img = ImageFactory.getInstance().fromFile(file);
        String json =
                "{\"image\": \""
                        + file.toUri().toURL()
                        + "\",\n"
                        + "\"prompt\": [\n"
                        + "    {\"type\": \"point\", \"data\": [575, 750], \"label\": 0},\n"
                        + "    {\"type\": \"rectangle\", \"data\": [425, 600, 700, 875]}\n"
                        + "]}";
        Sam2Input input = Sam2Input.fromJson(json);
        Assert.assertEquals(input.getPoints().size(), 1);
        Assert.assertEquals(input.getBoxes().size(), 1);

        input = Sam2Input.builder(img).addPoint(0, 1).addBox(0, 0, 1, 1).build();
        Assert.assertEquals(input.getPoints().size(), 1);
        Assert.assertEquals(input.getBoxes().size(), 1);
    }
}
