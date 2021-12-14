set VERSION="%1"
set FILEPATH="sentencepiece"

if exist %FILEPATH% (
    echo Found %FILEPATH%
) else (
    echo Couldn't find %FILEPATH%, downloading it ...
    echo Cloning google/sentencepiece.git
    git clone https://github.com/google/sentencepiece.git -b %VERSION%
    echo Finished cloning sentencepiece
)

if exist build rd /q /s build
md build\classes
cd build

javac -sourcepath ../src/main/java/ ../src/main/java/ai/djl/sentencepiece/jni/SentencePieceLibrary.java -h include -d classes
cmake ..
cmake --build . --config Release

:: for nightly ci
md jnilib\win-x86_64
copy Release\sentencepiece_native.dll jnilib\win-x86_64

cd ..
