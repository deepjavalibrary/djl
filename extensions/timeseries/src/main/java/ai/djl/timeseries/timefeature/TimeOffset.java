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

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** This is a class use to get multiple and granularity from frequency string. */
public class TimeOffset {

    private static String oPattern = "(?<multiple>\\d*)(?<granularity>\\w+)";

    private String name;
    private int n;

    /**
     * Constructs a new {@code DateOffset} instance.
     *
     * @param name offset granularity including "D", "M" etc
     * @param n offset multiple
     */
    public TimeOffset(String name, int n) {
        this.name = name;
        this.n = n;
    }

    /**
     * Return {@code TimeOffset} object from frequency string.
     *
     * @param freqStr Frequency string of the form [multiple][granularity] such as "12H", "1D" etc.
     * @return a TimeOffset containing multiple and granularity
     */
    public static TimeOffset toOffset(String freqStr) {
        Matcher matcher = Pattern.compile(oPattern).matcher(freqStr);
        matcher.find();
        String name = matcher.group("granularity");
        if ("".equals(name)) {
            throw new IllegalArgumentException("Invalid frequency");
        }
        name = "min".equals(name) ? "T" : name;

        String multiple = matcher.group("multiple");
        if ("".equals(multiple)) {
            multiple = "1";
        }
        int n = Integer.parseInt(multiple);
        return new TimeOffset(name, n);
    }

    /**
     * Return the granularity name of {@code TimeOffset}.
     *
     * @return the granularity name of {@code TimeOffset}.
     */
    public String getName() {
        return name;
    }

    /**
     * Return the multiple of {@code TimeOffset}.
     *
     * @return the multiple of {@code TimeOffset}
     */
    public double getN() {
        return n;
    }

    /**
     * Return the formatted frequency string.
     *
     * @return the formatted frequency string
     */
    public String toFreqStr() {
        return n + name;
    }
}
