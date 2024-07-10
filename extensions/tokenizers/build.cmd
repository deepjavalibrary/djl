@rem https://chocolatey.org/docs/installation#install-with-cmdexe
@rem to install rust java etc..
@rem choco install rust -y
@rem choco install jdk8 -y

set VERSION=v"%1"

if exist "tokenizers" (
    echo Found "tokenizers"
) else (
    git clone https://github.com/huggingface/tokenizers -b %VERSION%
)

if exist build rd /q /s build
md build\classes

set RUST_MANIFEST=rust/Cargo.toml
cargo build --manifest-path %RUST_MANIFEST% --release

@rem for nightly ci
md build\jnilib\win-x86_64\cpu
copy rust\target\release\djl.dll build\jnilib\win-x86_64\cpu\tokenizers.dll
