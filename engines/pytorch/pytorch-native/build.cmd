@ECHO OFF
@rem https://chocolatey.org/docs/installation#install-with-cmdexe
@rem to install cmake java etc..
@rem choco install cmake.install --installargs '"ADD_CMAKE_TO_PATH=User"' -y
@rem choco install zulu11 -y

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

if "%VERSION%" == "1.13.1" (
    set PT_VERSION=V1_13_X
)
if "%VERSION%" == "2.0.1" (
    set PT_VERSION=V1_13_X
)
if "%VERSION%" == "2.1.1" (
    set PT_VERSION=V1_13_X
)
if "%VERSION%" == "2.1.2" (
    set PT_VERSION=V1_13_X
)

if /i "%2:~0,2%" == "cu" (
    set USE_CUDA=1
)

copy /y src\main\patch\cuda.cmake libtorch\share\cmake\Caffe2\public\

@rem workaround VS 17.4.0 issue: https://stackoverflow.com/questions/74366357/updating-to-visual-studio-17-4-0-yields-linker-errors-related-to-tls
copy /y src\main\patch\%VERSION%\Parallel.h libtorch\include\ATen\

if exist build rd /q /s build
md build\classes
cd build
javac -sourcepath ..\..\pytorch-engine\src\main\java\ ..\..\pytorch-engine\src\main\java\ai\djl\pytorch\jni\PyTorchLibrary.java -h include -d classes
cmake -DCMAKE_PREFIX_PATH=libtorch -DPT_VERSION=%PT_VERSION% -DUSE_CUDA=%USE_CUDA% ..
cmake --build . --config Release
