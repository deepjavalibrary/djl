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
package ai.djl.training.util;

import ai.djl.util.Progress;

/**
 * {@code ProgressBar} is an implementation of {@link Progress}. It can be used to display the
 * progress of a task in the form a bar.
 */
public final class ProgressBar implements Progress {

    private static final int TOTAL_BAR_LENGTH = 40;

    private String message;
    private String trailingMessage;
    private long max;
    private long progress;
    private int currentPercent;
    private char progressChar = getProgressChar();

    /** Creates an instance of {@code ProgressBar} with a maximum value of 1. */
    public ProgressBar() {
        max = 1;
    }

    /**
     * Creates an instance of {@code ProgressBar} with the given maximum value, and displays the
     * given message.
     *
     * @param message the message to be displayed
     * @param max the maximum value
     */
    public ProgressBar(String message, long max) {
        reset(message, max);
    }

    /**
     * Creates an instance of {@code ProgressBar} with the given maximum value, and displays the
     * given message.
     *
     * @param message the message to be displayed
     * @param max the maximum value
     * @param trailingMessage the trailing message to be shown
     */
    public ProgressBar(String message, long max, String trailingMessage) {
        reset(message, max);
        this.trailingMessage = trailingMessage;
    }

    /** {@inheritDoc} */
    @Override
    public final void reset(String message, long max, String trailingMessage) {
        this.message = trimMessage(message);
        this.max = max;
        this.trailingMessage = trailingMessage;
        currentPercent = 0;
        progress = 0;
    }

    /** {@inheritDoc} */
    @Override
    public void start(long initialProgress) {
        update(initialProgress);
    }

    /** {@inheritDoc} */
    @Override
    public void end() {
        update(max - 1);
    }

    /** {@inheritDoc} */
    @Override
    public void increment(long increment) {
        update(progress + increment);
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.SystemPrintln")
    public void update(long progress, String additionalMessage) {
        if (Boolean.getBoolean("disableProgressBar") || max <= 1) {
            return;
        }

        this.progress = progress;
        if (additionalMessage == null) {
            additionalMessage = trailingMessage;
        }
        int percent = (int) ((progress + 1) * 100 / max);
        percent = Math.min(percent, 100);
        if (percent == currentPercent && percent > 0) {
            // no need to refresh
            return;
        }

        currentPercent = percent;
        StringBuilder sb = new StringBuilder(100);
        sb.append('\r').append(message).append(':');
        for (int i = 0; i < 12 - message.length(); ++i) {
            sb.append(' ');
        }
        sb.append(String.format("%3d", percent)).append("% |");
        for (int i = 0; i < TOTAL_BAR_LENGTH; ++i) {
            if (i <= percent * TOTAL_BAR_LENGTH / 100) {
                sb.append(progressChar);
            } else {
                sb.append(' ');
            }
        }
        sb.append('|');
        if (additionalMessage != null) {
            sb.append(' ').append(additionalMessage);
        }
        if (percent == 100) {
            System.out.println(sb);
        } else {
            System.out.print(sb);
        }
    }

    private String trimMessage(String message) {
        int len = message.length();
        if (len < 13) {
            return message;
        }
        return message.substring(0, 4) + "..." + message.substring(len - 5);
    }

    private static char getProgressChar() {
        if (System.getProperty("os.name").startsWith("Win")) {
            return '=';
        } else if (System.getProperty("os.name").startsWith("Linux")) {
            String lang = System.getenv("LANG");
            if (lang == null || !lang.contains("UTF-8")) {
                return '=';
            }
        }
        return '█';
    }
}
