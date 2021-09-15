@ECHO OFF
@rem https://chocolatey.org/docs/installation#install-with-cmdexe
@rem to install cmake java etc..
@rem choco install cmake.install --installargs '"ADD_CMAKE_TO_PATH=User"' -y
@rem choco install jdk8 -y

set FILEPATH="libtorch"
set VERSION="1.9.0"
if "%1" == "cpu" (
    set DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-%VERSION%%%2Bcpu.zip"
) else if "%1" == "cu102" (
      set DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-%VERSION%%%2Bcu102.zip"
) else if "%1" == "cu111" (
      set DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-%VERSION%%%2Bcu111.zip"
)


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

if exist build rd /q /s build
md build\classes
cd build
javac -sourcepath ..\..\pytorch-engine\src\main\java\ ..\..\pytorch-engine\src\main\java\ai\djl\pytorch\jni\PyTorchLibrary.java -h include -d classes
cmake -DCMAKE_PREFIX_PATH=libtorch ..
cmake --build . --config Release
