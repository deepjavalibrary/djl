# DJL - Integration Test

This folder contains Deep Java Library (DJL) tests that integrate with the engine.
These tests are used to check both the engine behavior and the other DJL modules.

These integration tests are run automatically by gradle when building the project using `./gradlew build` from the main directory. You can also run `../gradlew build` from the integration directory.

The integration tests are divided into two sections: the main tests and the nightly tests. The nightly tests include some longer running tests such as full training. The rest of the tests are intended to be much faster. By default, only the faster tests are run. The full nightly tests can be run with `./gradlew -Dnightly=true build`.

When running the integration tests, code coverage is also collected. The easiest way to view this coverage is by looking at the report in `build/reports/jacoco/test/html/index.html`.

## Switch Engine for tests
You can switch the engine through setting the system property `ai.djl.default_engine`:

```bash
./gradlew build -Dai.djl.default_engine=<Engine_Name>
```

### Windows PowerShell

```bash
..\gradlew build "-Dai.djl.default_engine=<Engine_Name>"
```
