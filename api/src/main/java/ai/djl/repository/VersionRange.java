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
package ai.djl.repository;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A {@code VersionRange} is a set of {@link Restriction}s that match some {@link Version}s.
 *
 * <p>A {@code VersionRange} should be constructed using {@link VersionRange#parse(String)}. The
 * format used by the version ranges matches the <a
 * href="https://cwiki.apache.org/confluence/display/MAVENOLD/Dependency+Mediation+and+Conflict+Resolution#DependencyMediationandConflictResolution-DependencyVersionRanges">maven
 * version range syntax</a>.
 */
public final class VersionRange {

    private static final VersionRange ANY = new VersionRange(null, Collections.emptyList());

    private Version recommendedVersion;
    private List<Restriction> restrictions;

    private VersionRange(Version recommendedVersion, List<Restriction> restrictions) {
        this.recommendedVersion = recommendedVersion;
        this.restrictions = restrictions;
    }

    /**
     * Returns the recommended version in the range.
     *
     * @return the recommended version in the range
     */
    public Version getRecommendedVersion() {
        return recommendedVersion;
    }

    /**
     * Returns the restrictions that compose the range.
     *
     * @return the restrictions that compose the range
     */
    public List<Restriction> getRestrictions() {
        return restrictions;
    }

    /**
     * Creates a new version range from a string version range.
     *
     * @param spec the string version range
     * @return the {@link VersionRange}
     */
    public static VersionRange parse(String spec) {
        if (spec == null || spec.isEmpty()) {
            return ANY;
        }

        List<Restriction> restrictions = new ArrayList<>();
        String process = spec;
        Version version = null;
        Version upperBound = null;
        Version lowerBound = null;

        while (process.startsWith("[") || process.startsWith("(")) {
            int index1 = process.indexOf(')');
            int index2 = process.indexOf(']');

            int index = index2;
            if (index2 < 0 || index1 < index2) {
                if (index1 >= 0) {
                    index = index1;
                }
            }

            if (index < 0) {
                throw new IllegalArgumentException("Unbounded range: " + spec);
            }

            Restriction restriction = parseRestriction(process.substring(0, index + 1));
            if (lowerBound == null) {
                lowerBound = restriction.getLowerBound();
            }
            if (upperBound != null) {
                if (restriction.getLowerBound() == null
                        || restriction.getLowerBound().compareTo(upperBound) < 0) {
                    throw new IllegalArgumentException("Ranges overlap: " + spec);
                }
            }
            restrictions.add(restriction);
            upperBound = restriction.getUpperBound();

            process = process.substring(index + 1).trim();

            if (process.length() > 0 && process.startsWith(",")) {
                process = process.substring(1).trim();
            }
        }

        if (process.length() > 0) {
            if (!restrictions.isEmpty()) {
                throw new IllegalArgumentException(
                        "Only fully-qualified sets allowed in multiple set scenario: " + spec);
            }

            version = new Version(process);
            restrictions.add(Restriction.EVERYTHING);
        }

        return new VersionRange(version, restrictions);
    }

    private static Restriction parseRestriction(String spec) {
        boolean lowerBoundInclusive = spec.startsWith("[");
        boolean upperBoundInclusive = spec.endsWith("]");

        String process = spec.substring(1, spec.length() - 1).trim();

        Restriction restriction;

        int index = process.indexOf(',');

        if (index < 0) {
            if (!lowerBoundInclusive || !upperBoundInclusive) {
                throw new IllegalArgumentException(
                        "Single version must be surrounded by []: " + spec);
            }

            Version version = new Version(process);

            restriction = new Restriction(version, true, version, true);
        } else {
            String lowerBound = process.substring(0, index).trim();
            String upperBound = process.substring(index + 1).trim();
            if (lowerBound.equals(upperBound)) {
                throw new IllegalArgumentException(
                        "Range cannot have identical boundaries: " + spec);
            }

            Version lowerVersion = null;
            if (lowerBound.length() > 0) {
                lowerVersion = new Version(lowerBound);
            }
            Version upperVersion = null;
            if (upperBound.length() > 0) {
                upperVersion = new Version(upperBound);
            }

            if (upperVersion != null
                    && lowerVersion != null
                    && upperVersion.compareTo(lowerVersion) < 0) {
                throw new IllegalArgumentException("Range defies version ordering: " + spec);
            }

            restriction =
                    new Restriction(
                            lowerVersion, lowerBoundInclusive, upperVersion, upperBoundInclusive);
        }

        return restriction;
    }

    /**
     * Filters the provided artifacts to those that match the version range.
     *
     * @param artifacts the artifacts to filter
     * @return the filtered artifacts
     */
    public List<Artifact> matches(List<Artifact> artifacts) {
        return artifacts.stream().filter(this::contains).collect(Collectors.toList());
    }

    /**
     * Returns true if a version falls within this range.
     *
     * @param version the version to test
     * @return true if the version falls within this range
     */
    public boolean contains(Version version) {
        if (ANY == this) {
            return true;
        }
        if (recommendedVersion != null) {
            return recommendedVersion.equals(version);
        }
        for (Restriction restriction : restrictions) {
            if (restriction.containsVersion(version)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Returns true if the artifact's version falls within this range.
     *
     * @param artifact the artifact to test
     * @return true if the artifact's version falls within this range
     */
    public boolean contains(Artifact artifact) {
        return artifact.getVersion() == null || contains(artifact.getParsedVersion());
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (recommendedVersion != null) {
            return recommendedVersion.toString();
        }

        StringBuilder buf = new StringBuilder();
        for (Iterator<Restriction> i = restrictions.iterator(); i.hasNext(); ) {
            Restriction r = i.next();

            buf.append(r.toString());

            if (i.hasNext()) {
                buf.append(',');
            }
        }
        return buf.toString();
    }
}
