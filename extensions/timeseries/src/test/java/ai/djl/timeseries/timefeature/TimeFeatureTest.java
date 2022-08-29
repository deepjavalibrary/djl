/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.timeseries.timefeature;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class TimeFeatureTest {

    @Test
    public void testGetLagsForFreq() {
        Map<String, List<Integer>> expectedLags = new ConcurrentHashMap<>();
        expectedLags.put(
                "4S",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 28, 29, 30, 31, 32, 43, 44, 45, 46,
                        47, 898, 899, 900, 901, 902));
        expectedLags.put(
                "min",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 58, 59, 60, 61, 62, 118, 119, 120, 121, 122, 178, 179,
                        180, 181, 182));
        expectedLags.put(
                "15min",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 95, 96, 97, 191, 192, 193,
                        287, 288, 289, 383, 384, 385, 479, 480, 481, 575, 576, 577, 671, 672, 673));
        expectedLags.put(
                "30min",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 8, 47, 48, 49, 95, 96, 97, 143, 144, 145, 191, 192,
                        193, 239, 240, 241, 287, 288, 289, 335, 336, 337, 671, 672, 673, 1007, 1008,
                        1009));
        expectedLags.put(
                "59min",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 47, 48, 49, 72, 73, 74, 96, 97, 98, 121,
                        122, 123, 145, 146, 147, 169, 170, 171, 340, 341, 342, 511, 512, 513, 682,
                        683, 684, 731, 732, 733));
        expectedLags.put(
                "61min",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 22, 23, 24, 46, 47, 48, 69, 70, 71, 93, 94, 95, 117,
                        118, 119, 140, 141, 142, 164, 165, 166, 329, 330, 331, 494, 495, 496, 659,
                        660, 661, 707, 708, 709));
        expectedLags.put(
                "H",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 47, 48, 49, 71, 72, 73, 95, 96, 97, 119,
                        120, 121, 143, 144, 145, 167, 168, 169, 335, 336, 337, 503, 504, 505, 671,
                        672, 673, 719, 720, 721));
        expectedLags.put(
                "6H",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25,
                        27, 28, 29, 55, 56, 57, 83, 84, 85, 111, 112, 113, 119, 120, 121, 224,
                        336));
        expectedLags.put(
                "12H",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 27, 28, 29, 41, 42, 43,
                        55, 56, 57, 59, 60, 61, 112, 168, 727, 728, 729));
        expectedLags.put(
                "23H",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 20, 21, 22, 28, 29, 30, 31, 32, 58, 87,
                        378, 379, 380, 758, 759, 760, 1138, 1139, 1140));
        expectedLags.put(
                "25H",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 19, 20, 21, 25, 26, 27, 28, 29, 53, 80,
                        348, 349, 350, 697, 698, 699, 1047, 1048, 1049));
        expectedLags.put(
                "D",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 20, 21, 22, 27, 28, 29, 30, 31, 56, 84,
                        363, 364, 365, 727, 728, 729, 1091, 1092, 1093));
        expectedLags.put(
                "2D",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 28, 42, 181, 182, 183,
                        363, 364, 365, 545, 546, 547));
        expectedLags.put(
                "6D",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 9, 14, 59, 60, 61, 120, 121, 122, 181, 182, 183));
        expectedLags.put(
                "W",
                Arrays.asList(
                        1, 2, 3, 4, 5, 6, 7, 8, 12, 51, 52, 53, 103, 104, 105, 155, 156, 157));
        expectedLags.put(
                "8D",
                Arrays.asList(1, 2, 3, 4, 5, 6, 7, 10, 44, 45, 46, 90, 91, 92, 135, 136, 137));
        expectedLags.put(
                "4W", Arrays.asList(1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 25, 26, 27, 38, 39, 40));
        expectedLags.put(
                "3W", Arrays.asList(1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 33, 34, 35, 51, 52, 53));
        expectedLags.put(
                "5W", Arrays.asList(1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 33, 34, 35, 51, 52, 53));
        expectedLags.put(
                "M", Arrays.asList(1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 23, 24, 25, 35, 36, 37));
        expectedLags.put("6M", Arrays.asList(1, 2, 3, 4, 5, 6, 7));
        expectedLags.put("12M", Arrays.asList(1, 2, 3, 4, 5, 6, 7));

        // For the default multiple (1)
        String[] defaultFreq = new String[]{"min", "H", "D", "W", "M"};
        for (String freq : defaultFreq) {
            expectedLags.put("1" + freq, expectedLags.get(freq));
        }

        // For frequencies that do not have unique form
        expectedLags.put("60min", expectedLags.get("1H"));
        expectedLags.put("24H", expectedLags.get("1D"));
        expectedLags.put("7D", expectedLags.get("1W"));

        String[] testFreqs = new String[]{"4S",
            "min",
            "1min",
            "15min",
            "30min",
            "59min",
            "60min",
            "61min",
            "H",
            "1H",
            "6H",
            "12H",
            "23H",
            "24H",
            "25H",
            "D",
            "1D",
            "2D",
            "6D",
            "7D",
            "8D",
            "W",
            "1W",
            "3W",
            "4W",
            "5W",
            "M",
            "6M",
            "12M"};
        for (String freq : testFreqs) {
            List<Integer> lags = Lag.getLagsForFreq(freq);
            if (!lags.equals(expectedLags.get(freq))) {
                System.out.println(freq);
            }
        }
    }
}
