/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package software.amazon.ai.ndarray.internal;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.Utils;

public abstract class NDFormat {

    private static final int MAX_DEPTH = 10;
    private static final int MAX_PRINT_ROWS = 10;
    private static final int MAX_PRINT_ITEMS = 20;
    private static final int PRECISION = 8;
    private static final String LF = System.getProperty("line.separator");
    private static final Pattern PATTERN = Pattern.compile("\\s*\\d\\.(\\d*?)0*e[+-](\\d+)");

    public static String format(NDArray array) {
        NDFormat format;
        DataType dataType = array.getDataType();

        if (dataType == DataType.UINT8) {
            format = new HexFormat();
        } else if (dataType.isInteger()) {
            format = new IntFormat(array);
        } else {
            format = new FloatFormat(array);
        }
        return format.dump(array);
    }

    protected abstract CharSequence format(Number value);

    private String dump(NDArray array) {
        StringBuilder sb = new StringBuilder(1000);
        sb.append("ND: ")
                .append(array.getShape())
                .append(' ')
                .append(array.getDevice())
                .append(' ')
                .append(array.getDataType())
                .append(LF);
        // corner case: 0 dimension
        if (array.size() == 0) {
            sb.append("[]").append(LF);
            return sb.toString();
        }
        // scalar case
        if (array.getShape().dimension() == 0) {
            sb.append(format(array.toArray()[0])).append(LF);
            return sb.toString();
        }
        if (array.getShape().dimension() < MAX_DEPTH) {
            dump(sb, array, 0, true);
        } else {
            sb.append("[ Exceed max print dimension ]");
        }
        return sb.toString();
    }

    private void dump(StringBuilder sb, NDArray array, int depth, boolean first) {
        if (!first) {
            Utils.pad(sb, ' ', depth);
        }
        sb.append('[');
        Shape shape = array.getShape();
        if (shape.dimension() == 1) {
            append(sb, array.toArray());
        } else {
            long len = shape.head();
            long limit = Math.min(len, MAX_PRINT_ROWS);
            for (int i = 0; i < limit; ++i) {
                try (NDArray nd = array.get(i)) {
                    dump(sb, nd, depth + 1, i == 0);
                }
            }
            long remaining = len - limit;
            if (remaining > 0) {
                Utils.pad(sb, ' ', depth + 1);
                sb.append("... ").append(remaining).append(" more");
            }
            Utils.pad(sb, ' ', depth);
        }
        // last "]"
        if (depth == 0) {
            sb.append(']').append(LF);
        } else {
            sb.append("],").append(LF);
        }
    }

    private void append(StringBuilder sb, Number[] values) {
        if (values.length == 0) {
            return;
        }
        long limit = Math.min(values.length, MAX_PRINT_ITEMS);
        sb.append(format(values[0]));
        for (int i = 1; i < limit; ++i) {
            sb.append(", ");
            sb.append(format(values[i]));
        }

        long remaining = values.length - limit;
        if (remaining > 0) {
            sb.append(", ... ").append(remaining).append(" more");
        }
    }

    private static final class FloatFormat extends NDFormat {

        private boolean exponential;
        private int precision;
        private int totalLength;

        public FloatFormat(NDArray array) {
            Number[] values = array.toArray();
            int maxIntPartLen = 0;
            int maxFractionLen = 0;
            int expFractionLen = 0;
            int maxExpSize = 2;
            boolean sign = false;

            double max = 0;
            double min = Double.MAX_VALUE;
            for (Number n : values) {
                double v = n.doubleValue();
                if (v < 0) {
                    sign = true;
                }

                if (!Double.isFinite(v)) {
                    int intPartLen = v < 0 ? 4 : 3;
                    if (totalLength < intPartLen) {
                        totalLength = intPartLen;
                    }
                    continue;
                }
                double abs = Math.abs(v);
                String str = String.format("%16e", abs);
                Matcher m = PATTERN.matcher(str);
                if (!m.matches()) {
                    throw new AssertionError("Invalid decimal value: " + str);
                }
                int fractionLen = m.group(1).length();
                if (expFractionLen < fractionLen) {
                    expFractionLen = fractionLen;
                }
                int expSize = m.group(2).length();
                if (expSize > maxExpSize) {
                    maxExpSize = expSize;
                }

                if (abs >= 1) {
                    int intPartLen = (int) Math.log10(abs) + 1;
                    if (v < 0) {
                        ++intPartLen;
                    }
                    if (intPartLen > maxIntPartLen) {
                        maxIntPartLen = intPartLen;
                    }
                    int fullFractionLen = fractionLen + 1 - intPartLen;
                    if (maxFractionLen < fullFractionLen) {
                        maxFractionLen = fullFractionLen;
                    }
                } else {
                    int intPartLen = v < 0 ? 2 : 1;
                    if (intPartLen > maxIntPartLen) {
                        maxIntPartLen = intPartLen;
                    }

                    int fullFractionLen = fractionLen + Integer.parseInt(m.group(2));
                    if (maxFractionLen < fullFractionLen) {
                        maxFractionLen = fullFractionLen;
                    }
                }

                if (abs > max) {
                    max = abs;
                }
                if (abs < min && abs > 0) {
                    min = abs;
                }
            }
            double ratio = max / min;
            if (max > 1.e8 || min < 0.0001 || ratio > 1000.) {
                exponential = true;
                precision = Math.min(PRECISION, expFractionLen);
                totalLength = precision + 4;
                if (sign) {
                    ++totalLength;
                }
            } else {
                precision = Math.min(4, maxFractionLen);
                int len = maxIntPartLen + precision + 1;
                if (totalLength < len) {
                    totalLength = len;
                }
            }
        }

        @Override
        public CharSequence format(Number value) {
            double d = value.doubleValue();
            if (Double.isNaN(d)) {
                return String.format("%" + totalLength + "s", "nan");
            } else if (Double.isInfinite(d)) {
                if (d > 0) {
                    return String.format("%" + totalLength + "s", "inf");
                } else {
                    return String.format("%" + totalLength + "s", "-inf");
                }
            }
            if (exponential) {
                precision = Math.max(PRECISION, precision);
                return String.format("% ." + precision + "e", value.doubleValue());
            }
            if (precision == 0) {
                String fmt = "%" + (totalLength - 1) + '.' + precision + "f.";
                return String.format(fmt, value.doubleValue());
            }

            String fmt = "%" + totalLength + '.' + precision + 'f';
            String ret = String.format(fmt, value.doubleValue());
            // Replace trailing zeros with space
            char[] chars = ret.toCharArray();
            for (int i = chars.length - 1; i >= 0; --i) {
                if (chars[i] == '0') {
                    chars[i] = ' ';
                } else {
                    break;
                }
            }
            return new String(chars);
        }
    }

    private static final class HexFormat extends NDFormat {

        @Override
        public CharSequence format(Number value) {
            return String.format("0x%02X", value.byteValue());
        }
    }

    private static final class IntFormat extends NDFormat {

        private boolean exponential;
        private int precision;
        private int totalLength;

        public IntFormat(NDArray array) {
            Number[] values = array.toArray();
            // scalar case
            if (values.length == 1) {
                totalLength = 1;
                return;
            }
            long max = 0;
            long negativeMax = 0;
            for (Number n : values) {
                long v = n.longValue();
                long abs = Math.abs(v);
                if (v < 0 && abs > negativeMax) {
                    negativeMax = abs;
                }
                if (abs > max) {
                    max = abs;
                }
            }

            if (max >= 1.e8) {
                exponential = true;
                precision = Math.min(PRECISION, (int) Math.log10(max) + 1);
            } else {
                int size = (int) Math.log10(max) + 1;
                int negativeSize = (int) Math.log10(negativeMax) + 2;
                totalLength = Math.max(size, negativeSize);
            }
        }

        @Override
        public CharSequence format(Number value) {
            if (exponential) {
                return String.format("% ." + precision + "e", value.floatValue());
            }
            return String.format("%" + totalLength + "d", value.longValue());
        }
    }
}
