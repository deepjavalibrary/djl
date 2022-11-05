@ECHO OFF
@rem https://chocolatey.org/docs/installation#install-with-cmdexe
@rem to install cmake java etc..
@rem choco install cmake.install --installargs '"ADD_CMAKE_TO_PATH=User"' -y
@rem choco install jdk8 -y

set FILEPATH="libtorch"
set VERSION=%1
set DOWNLOAD_URL="https://download.pytorch.org/libtorch/%2/libtorch-win-shared-with-deps-%VERSION%%%2B%2.zip"

if exist %FILEPATH% (
    echo Found %FILEPATH%
) else (
    echo Couldn't find %FILEPATH%, downloading it ...
    echo Downloading from: %DOWNLOAD_URL%
    powershell -Command "(New-Object Net.WebClient).DownloadFile('%DOWNLOAD_URL%', '%cd%\libtorch.zip')"
    powershell -Command "Expand-Archive -LiteralPath libtorch.zip -DestinationPath %cd%"
    del /f libtorch.zip
    echo Finished downloading libtorch
)

copy /y src\main\patch\cuda.cmake libtorch\share\cmake\Caffe2\public\

if exist build rd /q /s build
md build\classes
cd build
javac -sourcepath ..\..\pytorch-engine\src\main\java\ ..\..\pytorch-engine\src\main\java\ai\djl\pytorch\jni\PyTorchLibrary.java -h include -d classes
cmake -DCMAKE_PREFIX_PATH=libtorch ..
cmake --build . --config Release
