@ECHO OFF
@rem https://chocolatey.org/docs/installation#install-with-cmdexe
@rem to install cmake java etc..
@rem choco install cmake.install --installargs '"ADD_CMAKE_TO_PATH=User"' -y
@rem choco install jdk8 -y

set FILEPATH="libtorch"
set VERSION=%1
if "%2" == "cpu" (
    set DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-%VERSION%%%2Bcpu.zip"
) else if "%2" == "cu102" (
      set DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-%VERSION%%%2Bcu102.zip"
) else if "%2" == "cu111" (
      set DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-%VERSION%%%2Bcu111.zip"
) else if "%2" == "cu113" (
      set DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-%VERSION%%%2Bcu113.zip"
) else if "%2" == "cu116" (
      set DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu116/libtorch-win-shared-with-deps-%VERSION%%%2Bcu116.zip"
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

if "%VERSION%" == "1.12.1" (
    copy /y src\main\patch\cuda.cmake libtorch\share\cmake\Caffe2\public\
)
if "%VERSION%" == "1.11.0" (
    copy /y src\main\patch\cuda.cmake libtorch\share\cmake\Caffe2\public\
)
if "%VERSION%" == "1.10.0" (
    set PT_OLD_VERSION=1
)
if "%VERSION%" == "1.9.1" (
    set PT_OLD_VERSION=1
)

if exist build rd /q /s build
md build\classes
cd build
javac -sourcepath ..\..\pytorch-engine\src\main\java\ ..\..\pytorch-engine\src\main\java\ai\djl\pytorch\jni\PyTorchLibrary.java -h include -d classes
cmake -DCMAKE_PREFIX_PATH=libtorch -DPT_OLD_VERSION=%PT_OLD_VERSION% ..
cmake --build . --config Release
