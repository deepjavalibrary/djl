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

import ai.djl.modality.Input;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.serving.wlm.ModelInfo;
import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * converts any byte-array into to required InputType.
 *
 * @author erik.bamberg@web.de
 */
public class InputTypeConverter {

    /**
     * convert to ByteArray from the request-input to the type expected by the model.
     *
     * @param model to model to convert for.
     * @param input The original input.
     * @return input-object in the correct Type.
     * @throws ConversionException exception when conversion fails.
     */
    public Object convertToInputData(ModelInfo model, Input input) throws ConversionException {
        try {
            if (model.getInputType() != null) {
                if (Image.class.isAssignableFrom(model.getInputType())) {
                    return ImageFactory.getInstance()
                            .fromInputStream(
                                    new ByteArrayInputStream(input.getContent().get("data")));
                }
            }
        } catch (IOException e) {
            throw new ConversionException(
                    "unable to convert "
                            + input.getClass().getName()
                            + " to "
                            + model.getInputType().getName(),
                    e);
        }

        return input;
    }
}
