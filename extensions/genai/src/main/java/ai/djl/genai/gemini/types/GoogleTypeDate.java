/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.genai.gemini.types;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class GoogleTypeDate {

    private Integer day;
    private Integer month;
    private Integer year;

    GoogleTypeDate(Builder builder) {
        day = builder.day;
        month = builder.month;
        year = builder.year;
    }

    public Integer getDay() {
        return day;
    }

    public Integer getMonth() {
        return month;
    }

    public Integer getYear() {
        return year;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code GoogleTypeDate}. */
    public static final class Builder {

        Integer day;
        Integer month;
        Integer year;

        public Builder day(Integer day) {
            this.day = day;
            return this;
        }

        public Builder month(Integer month) {
            this.month = month;
            return this;
        }

        public Builder year(Integer year) {
            this.year = year;
            return this;
        }

        public GoogleTypeDate build() {
            return new GoogleTypeDate(this);
        }
    }
}
