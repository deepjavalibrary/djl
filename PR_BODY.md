## Description ##

Adds configuration controls around two loading behaviors and tightens their defaults. Every previous behavior remains available, but from this release **it has to be enabled explicitly** — the defaults changed, so an existing deployment that relies on one of the behaviors below needs to set the corresponding flag.

**1. Compiling bundled `.java` sources at model load is now opt-in**

`ClassLoaderUtils.compileJavaClass` previously compiled and loaded any `.java` sources found under a model's `lib/classes/` directory whenever a model was loaded. This now requires an explicit opt-in. When sources are present and the opt-in is absent, the load fails with a message naming the flag rather than continuing — the caller resolves a translator from the compiled output, so continuing would silently fall back to a default translator and change inference results with only a log line to explain it.

**2. Remote URL loading is limited to public http(s) destinations**

`Utils.openUrl` now:
- allows `http`/`https` to public destinations;
- resolves redirects explicitly (rather than relying on `HttpURLConnection` auto-follow) and applies the same destination rule to each hop, with a bounded hop count;
- drops `Authorization`, `Cookie` and `Proxy-Authorization` when a redirect crosses to a different scheme, host or port;
- leaves local-resource schemes (`file:`, `jar:`) unrestricted, since they perform no network fetch;
- treats other schemes as unsupported.

**3. `SimpleUrlRepository` and `AbstractRepository` share the same URL handling**

`SimpleUrlRepository.download()`, its content-length `HEAD` probe, and `AbstractRepository.download()` (for absolute artifact URIs declared in repository metadata) previously issued their own connections. All three now go through the shared entry point, so the download paths and the probe behave consistently.

## What changes for existing deployments ##

| Previously worked by default | Now | To keep it working |
|---|---|---|
| Model archive shipping a raw `.java` translator under `lib/classes/` | fails at load with a message naming the flag | `DJL_COMPILE_JAVA=true` or `-Dai.djl.compile_java=true` |
| `http(s)` load from a host on a private network — an internal artifact mirror on an RFC 1918 address, a Kubernetes in-cluster service address, a VPC endpoint reached by private DNS, or a `localhost` server | fails with `Blocked request to non-public address: <host>` | `DJL_ALLOW_INSECURE_URL=true` or `-Dai.djl.allow_insecure_url=true` |
| `ftp:` or another custom `URLStreamHandler` scheme passed to `Utils.openUrl` | fails with `Blocked request using unsupported URL protocol: <scheme>` | same flag as above |
| A URL behind more than 5 redirect hops (the JVM default was 20) | fails with `Too many redirects` | same flag as above |
| A public URL that redirects to a private address | fails on the redirect hop | same flag as above |
| Caller-supplied credential headers carried across a cross-origin redirect | headers dropped on that hop | same flag as above |

`DJL_ALLOW_INSECURE_URL` is deliberately a single all-or-nothing switch: setting it restores the entire previous path (no destination rule, no explicit redirect resolution, no scheme restriction, no header handling). There is no per-host allowlist. If that turns out to be too coarse for real deployments — for example an operator who wants one internal mirror reachable without disabling the rest — a narrower `DJL_ALLOWED_HOSTS`-style control would be the natural follow-up, and I'm happy to add it here instead if you'd prefer.

Both flags follow the existing `DJL_OFFLINE` / `ai.djl.offline` convention (environment variable first, then system property). `docs/create_serving_ready_model.md` documented the bundled-`.java` path as a supported way to ship a translator and has been updated to record the opt-in.

## Not affected ##

No API changes to existing methods; one new public method (`Utils.openHttpConnection`) so callers that only need response metadata share the same handling.

These loading flows behave exactly as before:

- local model directories and `file:` paths (`LocalRepository`);
- `s3://` and `gs://` (handled by their own `RepositoryFactory` implementations, which never call `openUrl`);
- public `http(s)` archives, the `djl://` model zoo, and Hugging Face;
- `jar:` classpath resources — including the way djl-serving reads each plugin's `plugin.definition`;
- engine native-library downloads, whose URLs are fixed to `https://publish.djl.ai/...`;
- models shipping precompiled `.class` or `.jar` translators, or supplying one programmatically via `Criteria.optTranslator(...)`.

## Testing ##

- New `UrlAccessAndCompilationTest` (21 tests) and `RepositoryDownloadTest` (6 tests) covering the new defaults, both flags, explicit redirect resolution against a real local HTTP server (302 followed, disallowed destination, unsupported scheme, missing `Location`, hop limit), the local-scheme cases, and the repository download paths.
- Negative controls: reverting each change with the tests in place makes the corresponding tests fail.
- `verifyJava`, `checkstyleMain/Test`, `pmdMain/Test`, `javadoc` all pass; all modules compile.
- Verified end-to-end in a djl-serving container: model loads and serves normally, all bundled plugins load, and inference throughput is unchanged (within run-to-run noise).
