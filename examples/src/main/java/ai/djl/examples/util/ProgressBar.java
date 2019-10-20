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
package ai.djl.examples.util;

public final class ProgressBar {

    private String message;
    private int max;
    private int current;

    public ProgressBar(String message, int max) {
        this.message = message;
        this.max = max;
    }

    @SuppressWarnings("PMD.SystemPrintln")
    public void printProgress(int index) {
        if (Boolean.getBoolean("disableProgressBar")) {
            return;
        }

        int percent = (index + 1) * 100 / max;
        if (percent == current) {
            return;
        }

        current = percent;
        StringBuilder sb = new StringBuilder(message);
        sb.append(':');
        for (int i = 0; i < 12 - message.length(); ++i) {
            sb.append(' ');
        }
        sb.append(String.format("%3d", percent)).append("% |");
        for (int i = 0; i < 60; ++i) {
            if (i <= percent * 60 / 100) {
                sb.append('█');
            } else {
                sb.append(' ');
            }
        }
        sb.append("|\r");
        if (index == max - 1) {
            System.out.println(sb);
        } else {
            System.out.print(sb);
        }
    }
}
