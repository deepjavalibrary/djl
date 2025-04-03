@rem https://chocolatey.org/docs/installation#install-with-cmdexe
@rem to install rust java etc..
@rem choco install rust -y
@rem choco install jdk8 -y

if exist build rd /q /s build
md build\classes

set RUST_MANIFEST=rust/Cargo.toml
cargo build --manifest-path %RUST_MANIFEST% --release

@rem for nightly ci
md build\jnilib\win-x86_64\cpu
copy rust\target\release\djl_tokenizer.dll build\jnilib\win-x86_64\cpu\tokenizers.dll
