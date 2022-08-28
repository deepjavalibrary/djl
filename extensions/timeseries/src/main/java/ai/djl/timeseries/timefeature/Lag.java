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

import java.util.ArrayList;
import java.util.Arrays;
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
     * @param freqStr Frequency string of the form [multiple][granularity] such as "12H", "1D", "6T"
     *     etc.
     * @param lagUb The maximum value for a lag
     * @return a list of lags
     */
    public static List<Integer> getLagsForFreq(String freqStr, int lagUb) {
        List<List<Integer>> lags;
        TimeOffset timeOffset = TimeOffset.toOffset(freqStr);
        switch (timeOffset.getName()) {
            case "Q":
                if (timeOffset.getN() != 1) {
                    throw new IllegalArgumentException(
                            "Only multiple 1 is supported for quarterly. Use x month instead.");
                }
                lags = makeLagsForMonth(3. * timeOffset.getN());
                break;
            case "M":
                lags = makeLagsForMonth(timeOffset.getN());
                break;
            case "W":
                lags = makeLagsForWeek(timeOffset.getN());
                break;
            case "D":
                lags = makeLagsForDay(timeOffset.getN());
                lags.addAll(makeLagsForWeek(timeOffset.getN() / 7.));
                break;
            case "H":
                lags = makeLagsForHour(timeOffset.getN());
                lags.addAll(makeLagsForDay(timeOffset.getN() / 24.));
                lags.addAll(makeLagsForWeek((timeOffset.getN() / (24. * 7.))));
                break;
            case "min":
            case "T":
                lags = makeLagsForMinute(timeOffset.getN());
                lags.addAll(makeLagsForHour(timeOffset.getN() / 60.));
                lags.addAll(makeLagsForDay(timeOffset.getN() / (60. * 24.)));
                lags.addAll(makeLagsForWeek(timeOffset.getN() / (60. * 24. * 7.)));
                break;
            case "S":
                lags = makeLagsForSecond(timeOffset.getN());
                lags.addAll(makeLagsForMinute(timeOffset.getN() / 60.));
                lags.addAll(makeLagsForHour(timeOffset.getN() / (60. * 60.)));
                break;
            default:
                throw new IllegalArgumentException("invalid frequency");
        }
        List<Integer> ret = new ArrayList<>();
        for (List<Integer> subList : lags) {
            for (Integer lag : subList) {
                if (lag > 7 && lag <= lagUb) {
                    ret.add(lag);
                }
            }
        }
        ret = ret.stream().distinct().collect(Collectors.toList());
        ret.sort(Comparator.naturalOrder());
        List<Integer> list = new ArrayList<>();
        for (int i = 1; i < 8; i++) {
            list.add(i);
        }
        ret.addAll(0, list);
        return ret;
    }

    /**
     * Generates a list of lags that are appropriate for the given frequency string. Set the lagUb
     * to default value 1200.
     *
     * @param freqStr Frequency string of the form [multiple][granularity] such as "12H", "1D" etc
     * @return a list of lags
     */
    public static List<Integer> getLagsForFreq(String freqStr) {
        return getLagsForFreq(freqStr, 1200);
    }

    private static List<List<Integer>> makeLagsForSecond(double multiple) {
        int numCycles = 3;
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i < numCycles + 1; i++) {
            ret.add(makeLags((int) (i * 60 / multiple), 2));
        }
        return ret;
    }

    private static List<List<Integer>> makeLagsForMinute(double multiple) {
        int numCycles = 3;
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i < numCycles + 1; i++) {
            ret.add(makeLags((int) (i * 60 / multiple), 2));
        }
        return ret;
    }

    private static List<List<Integer>> makeLagsForHour(double multiple) {
        int numCycles = 7;
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i < numCycles + 1; i++) {
            ret.add(makeLags((int) (i * 24 / multiple), 1));
        }
        return ret;
    }

    private static List<List<Integer>> makeLagsForDay(double multiple) {
        int numCycles = 4;
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i < numCycles + 1; i++) {
            ret.add(makeLags((int) (i * 7 / multiple), 1));
        }
        ret.add(makeLags((int) (30 / multiple), 1));
        return ret;
    }

    private static List<List<Integer>> makeLagsForWeek(double multiple) {
        int numCycles = 3;
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i < numCycles + 1; i++) {
            ret.add(makeLags((int) (i * 52 / multiple), 1));
        }
        ret.add(
                Arrays.asList(
                        (int) (4. / multiple), (int) (8. / multiple), (int) (12. / multiple)));
        return ret;
    }

    private static List<List<Integer>> makeLagsForMonth(double multiple) {
        int numCycles = 3;
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i < numCycles + 1; i++) {
            ret.add(makeLags((int) (i * 12 / multiple), 1));
        }
        return ret;
    }

    /**
     * Creates a set of lags around a middle point including +/- delta.
     *
     * @param middle the middle point
     * @param delta the delta
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
