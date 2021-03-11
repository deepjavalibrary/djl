/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.nn.convolutional;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;

/** Utility class for calculating output shapes of convolution blocks. */
public abstract class ShapeUtils {

    private ShapeUtils() {}

    /**
     * Calculate the output shape for a convolution block.
     *
     * @param manager the manager used to calculate the output shape
     * @param block the block whose output shape should be calculated
     * @param inputShape the shape for the block to use as it's input
     * @return the corresponding output shape for the provided input
     */
    public static Shape outputShapeForBlock(NDManager manager, Block block, Shape inputShape) {
        Shape[] outputs = block.getOutputShapes(new Shape[] {inputShape});
        return outputs[0];
    }

    /**
     * Calculate the output dimension of a convolution. See <a href="
     * https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html">here</a> for further
     * information.
     *
     * <ul>
     *   <li>{@code input shape: (batch_size, channel, dimension1, dimension2 ... dimensionN)}
     *   <li>{@code output shape: (batch_size, num_filter, out_dimension1, out_dimension2 ...
     *       out_dimensionN)} <br>
     *       {@code out_dimension1 = f(dimension1, kernel[0], pad[0], stride[0], dilate[0])} <br>
     *       {@code out_dimension1 = f(dimension2, kernel[1], pad[1], stride[1], dilate[1])} <br>
     *       {@code out_dimensionN = f(dimensionN, kernel[N - 1], pad[N - 1], stride[N - 1],
     *       dilate[N - 1])} <br>
     *       {@code where f(x, k, p, s, d) = floor((x + 2 * p - d * (k - 1) - 1) / s) + 1}
     * </ul>
     *
     * @param dimension the input dimension
     * @param kernelComponent the kernel component in the provided dimension
     * @param padComponent the padding component in the provided dimension
     * @param strideComponent the stride component in the provided dimension
     * @param dilationComponent the dilation component in the provided dimension
     * @return the output dimension
     */
    public static long convolutionDimensionCalculation(
            long dimension,
            long kernelComponent,
            long padComponent,
            long strideComponent,
            long dilationComponent) {
        return ((dimension + 2 * padComponent - dilationComponent * (kernelComponent - 1) - 1)
                        / strideComponent)
                + 1;
    }

    /**
     * Calculate the output dimension of a deconvolution. See <a href="
     * https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d">here</a>
     * for further information.
     *
     * <ul>
     *   <li>{@code input shape: (batch_size, channel, dimension1, dimension2 ... dimensionN)}
     *   <li>{@code output shape: (batch_size, num_filter, out_dimension1, out_dimension2 ...
     *       out_dimensionN)} <br>
     *       {@code out_dimension1 = f(dimension1, kernel[0], pad[0], outputPad[0], stride[0],
     *       dilate[0])} <br>
     *       {@code out_dimension1 = f(dimension2, kernel[1], pad[1], outputPad[0], stride[1],
     *       dilate[1])} <br>
     *       {@code out_dimensionN = f(dimensionN, kernel[N - 1], pad[N - 1], outputPad[N - 1],
     *       stride[N - 1], dilate[N - 1])} <br>
     *       {@code where f(x, k, p, oP, s, d) = (x - 1) * s - 2 * p + d * (k - 1) + oP + 1}
     * </ul>
     *
     * @param dimension the input dimension
     * @param kernelComponent the kernel component in the provided dimension
     * @param padComponent the padding component in the provided dimension
     * @param outputPadComponent the output padding component in the provided dimension
     * @param strideComponent the stride component in the provided dimension
     * @param dilationComponent the dilation component in the provided dimension
     * @return the output dimension
     */
    public static long deconvolutionDimensionCalculation(
            long dimension,
            long kernelComponent,
            long padComponent,
            long outputPadComponent,
            long strideComponent,
            long dilationComponent) {
        return (dimension - 1) * strideComponent
                - 2 * padComponent
                + dilationComponent * (kernelComponent - 1)
                + outputPadComponent
                + 1;
    }
}
