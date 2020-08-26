# Update the VARIANT arg in devcontainer.json to pick a Java version: 8, 11, 14
ARG VARIANT=11
FROM openjdk:${VARIANT}-jdk-buster

# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="false"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
COPY library-scripts/common-debian.sh /tmp/library-scripts/
RUN bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" \
    && if [ ! -d "/docker-java-home" ]; then ln -s "${JAVA_HOME}" /docker-java-home; fi \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

# Install SDKMAN and optionally Maven and Gradle
ARG INSTALL_MAVEN="false"
ARG MAVEN_VERSION=3.6.3
ARG INSTALL_GRADLE="false"
ARG GRADLE_VERSION=5.4.1
ENV SDKMAN_DIR="/usr/local/sdkman"
ENV PATH="${PATH}:${SDKMAN_DIR}/java/current/bin:${SDKMAN_DIR}/maven/current/bin:${SDKMAN_DIR}/gradle/current/bin"
COPY library-scripts/java-debian.sh /tmp/library-scripts/
RUN bash /tmp/library-scripts/java-debian.sh "none" "${SDKMAN_DIR}" "${USERNAME}" "true" \
    && if [ "${INSTALL_MAVEN}" = "true" ]; then bash /tmp/library-scripts/maven-debian.sh ${MAVEN_VERSION} "${SDKMAN_DIR}" ${USERNAME} "true"; fi \
    && if [ "${INSTALL_GRADLE}" = "true" ]; then bash /tmp/library-scripts/gradle-debian.sh ${GRADLE_VERSION} "${SDKMAN_DIR}" ${USERNAME} "true"; fi \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

# [Optional] Install Node.js for use with web applications - update the INSTALL_NODE arg in devcontainer.json to enable.
ARG INSTALL_NODE="true"
ARG NODE_VERSION="none"
ENV NVM_DIR=/usr/local/share/nvm
ENV NVM_SYMLINK_CURRENT=true \
    PATH="${NVM_DIR}/current/bin:${PATH}"
COPY library-scripts/node-debian.sh /tmp/library-scripts/
RUN if [ "$INSTALL_NODE" = "true" ]; then bash /tmp/library-scripts/node-debian.sh "${NVM_DIR}" "${NODE_VERSION}" "${USERNAME}"; fi \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>