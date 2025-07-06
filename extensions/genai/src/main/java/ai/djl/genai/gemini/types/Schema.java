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

import com.google.gson.annotations.SerializedName;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Schema {

    private List<Schema> anyOf;

    @SerializedName("default")
    private Object defaultName;

    private String description;

    @SerializedName("enum")
    private List<String> enumName;

    private Object example;
    private String format;
    private Schema items;
    private Long maxItems;
    private Long maxLength;
    private Long maxProperties;
    private Double maximum;
    private Long minItems;
    private Long minLength;
    private Long minProperties;
    private Double minimum;
    private Boolean nullable;
    private String pattern;
    private Map<String, Schema> properties;
    private List<String> propertyOrdering;
    private List<String> required;
    private String title;
    private Type type;

    Schema(Builder builder) {
        anyOf = builder.anyOf;
        defaultName = builder.defaultName;
        description = builder.description;
        enumName = builder.enumName;
        example = builder.example;
        format = builder.format;
        items = builder.items;
        maxItems = builder.maxItems;
        maxLength = builder.maxLength;
        maxProperties = builder.maxProperties;
        maximum = builder.maximum;
        minItems = builder.minItems;
        minLength = builder.minLength;
        minProperties = builder.minProperties;
        minimum = builder.minimum;
        nullable = builder.nullable;
        pattern = builder.pattern;
        properties = builder.properties;
        propertyOrdering = builder.propertyOrdering;
        required = builder.required;
        title = builder.title;
        type = builder.type;
    }

    public List<Schema> getAnyOf() {
        return anyOf;
    }

    public Object getDefaultName() {
        return defaultName;
    }

    public String getDescription() {
        return description;
    }

    public List<String> getEnumName() {
        return enumName;
    }

    public Object getExample() {
        return example;
    }

    public String getFormat() {
        return format;
    }

    public Schema getItems() {
        return items;
    }

    public Long getMaxItems() {
        return maxItems;
    }

    public Long getMaxLength() {
        return maxLength;
    }

    public Long getMaxProperties() {
        return maxProperties;
    }

    public Double getMaximum() {
        return maximum;
    }

    public Long getMinItems() {
        return minItems;
    }

    public Long getMinLength() {
        return minLength;
    }

    public Long getMinProperties() {
        return minProperties;
    }

    public Double getMinimum() {
        return minimum;
    }

    public Boolean getNullable() {
        return nullable;
    }

    public String getPattern() {
        return pattern;
    }

    public Map<String, Schema> getProperties() {
        return properties;
    }

    public List<String> getPropertyOrdering() {
        return propertyOrdering;
    }

    public List<String> getRequired() {
        return required;
    }

    public String getTitle() {
        return title;
    }

    public Type getType() {
        return type;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code Schema}. */
    public static final class Builder {

        List<Schema> anyOf = new ArrayList<>();
        Object defaultName;
        String description;
        List<String> enumName;
        Object example;
        String format;
        Schema items;
        Long maxItems;
        Long maxLength;
        Long maxProperties;
        Double maximum;
        Long minItems;
        Long minLength;
        Long minProperties;
        Double minimum;
        Boolean nullable;
        String pattern;
        Map<String, Schema> properties;
        List<String> propertyOrdering;
        List<String> required;
        String title;
        Type type;

        public Builder anyOf(List<Schema> anyOf) {
            this.anyOf.clear();
            this.anyOf.addAll(anyOf);
            return this;
        }

        public Builder addAnyOf(Schema anyOf) {
            this.anyOf.add(anyOf);
            return this;
        }

        public Builder addAnyOf(Schema.Builder anyOf) {
            this.anyOf.add(anyOf.build());
            return this;
        }

        public Builder defaultName(Object defaultName) {
            this.defaultName = defaultName;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder enumName(List<String> enumName) {
            this.enumName = enumName;
            return this;
        }

        public Builder example(Object example) {
            this.example = example;
            return this;
        }

        public Builder format(String format) {
            this.format = format;
            return this;
        }

        public Builder items(Schema items) {
            this.items = items;
            return this;
        }

        public Builder items(Schema.Builder items) {
            this.items = items.build();
            return this;
        }

        public Builder maxItems(Long maxItems) {
            this.maxItems = maxItems;
            return this;
        }

        public Builder maxLength(Long maxLength) {
            this.maxLength = maxLength;
            return this;
        }

        public Builder maxProperties(Long maxProperties) {
            this.maxProperties = maxProperties;
            return this;
        }

        public Builder maximum(Double maximum) {
            this.maximum = maximum;
            return this;
        }

        public Builder minItems(Long minItems) {
            this.minItems = minItems;
            return this;
        }

        public Builder minLength(Long minLength) {
            this.minLength = minLength;
            return this;
        }

        public Builder minProperties(Long minProperties) {
            this.minProperties = minProperties;
            return this;
        }

        public Builder minimum(Double minimum) {
            this.minimum = minimum;
            return this;
        }

        public Builder nullable(Boolean nullable) {
            this.nullable = nullable;
            return this;
        }

        public Builder pattern(String pattern) {
            this.pattern = pattern;
            return this;
        }

        public Builder properties(Map<String, Schema> properties) {
            this.properties = properties;
            return this;
        }

        public Builder propertyOrdering(List<String> propertyOrdering) {
            this.propertyOrdering = propertyOrdering;
            return this;
        }

        public Builder required(List<String> required) {
            this.required = required;
            return this;
        }

        public Builder title(String title) {
            this.title = title;
            return this;
        }

        public Builder type(Type type) {
            this.type = type;
            return this;
        }

        public Schema build() {
            return new Schema(this);
        }
    }
}
