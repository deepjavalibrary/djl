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

package ai.djl.timeseries.timeFeature;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/** This class contains static method for get lags from frequency. */
public final class Lag {

    private Lag() {}

    /**
     * Generates a list of lags that are appropriate for the given frequency string.
     *
     * <p>By default all frequencies have the following lags: [1, 2, 3, 4, 5, 6, 7]. Remaining lags
     * correspond to the same `season` (+/- `delta`) in previous `k` cycles. Here `delta` and `k`
     * are chosen according to the existing code.
     *
     * @param freqStr Frequency string of the form [multiple][granularity] such as "12H", "1D" etc.
     * @param lagUb The maximum value for a lag
     * @return a list of lags
     */
    public static List<Integer> getLagsForFreq(String freqStr, int lagUb) {
        List<List<Integer>> lags;
        if ("D".equals(freqStr)) {
            lags = makeLagsForDay(1);
            lags.addAll(makeLagsForWeek(1f / 7f));
        } else {
            throw new IllegalArgumentException("invalid frequency: freq now must be D");
        }
        List<Integer> ret =
                new ArrayList<Integer>() {
                    {
                        for (List<Integer> subList : lags) {
                            for (Integer lag : subList) {
                                if (lag > 7 && lag <= lagUb) {
                                    add(lag);
                                }
                            }
                        }
                    }
                };
        ret = ret.stream().distinct().collect(Collectors.toList());
        ret.sort(Comparator.naturalOrder());
        ret.addAll(
                0,
                new ArrayList<Integer>() {
                    {
                        for (int i = 1; i < 8; i++) {
                            add(i);
                        }
                    }
                });
        return ret;
    }

    /**
     * Generates a list of lags that are appropriate for the given frequency string. Set the lagUb
     * to default value 1200.
     *
     * @param freqStr Frequency string of the form [multiple][granularity] such as "12H", "1D" etc.
     * @return a list of lags
     */
    public static List<Integer> getLagsForFreq(String freqStr) {
        return getLagsForFreq(freqStr, 1200);
    }

    private static List<List<Integer>> makeLagsForDay(float multiple) {
        int numCycles = 4;
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i < numCycles + 1; i++) {
            ret.add(makeLags((int) (i * 7 / multiple), 1));
        }
        ret.add(makeLags((int) (30 / multiple), 1));
        return ret;
    }

    private static List<List<Integer>> makeLagsForWeek(float multiple) {
        int numCycles = 3;
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i < numCycles + 1; i++) {
            ret.add(makeLags((int) (i * 52 / multiple), 1));
        }
        ret.add(
                new ArrayList<Integer>() {
                    {
                        add((int) (4f / multiple));
                        add((int) (8f / multiple));
                        add((int) (12f / multiple));
                    }
                });
        return ret;
    }

    /**
     * Create a set of lags around a middle point including +/- delta.
     *
     * @param middle middle point
     * @param delta +/- delta.
     * @return a range between [middle - delta, middle + delta]
     */
    private static List<Integer> makeLags(int middle, int delta) {
        List<Integer> ret = new ArrayList<>();
        for (int i = middle - delta; i < middle + delta + 1; i++) {
            ret.add(i);
        }
        return ret;
    }
}
