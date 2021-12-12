set VERSION="v0.1.95"
set FILEPATH="sentencepiece"

if exist %FILEPATH% (
    echo Found %FILEPATH%
) else (
    echo Couldn't find %FILEPATH%, downloading it ...
    echo Cloning google/sentencepiece.git
    powershell -Command "git clone https://github.com/google/sentencepiece.git -b %VERSION%"
    echo Finished cloning sentencepiece
)

if not exist build (
    mkdir build
)

cd build

rd classes
md classes
javac -sourcepath ../src/main/java/ ../src/main/java/ai/djl/sentencepiece/jni/SentencePieceLibrary.java -h include -d classes
cmake ..
cmake --build . --config Release

popd
@ECHO OFF
:: for nightly ci
md build/jnilib/win-x86_64
copy build/libsentencepiece_native.dll build/jnilib/win-x86_64/
