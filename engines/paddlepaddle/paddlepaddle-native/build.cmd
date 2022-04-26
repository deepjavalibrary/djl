@ECHO OFF
@rem https://chocolatey.org/docs/installation#install-with-cmdexe
@rem to install cmake java etc..
@rem choco install cmake.install --installargs '"ADD_CMAKE_TO_PATH=User"' -y
@rem choco install jdk8 -y

set FILEPATH="paddle"

if "%1" == "cpu" (
  set DOWNLOAD_URL="https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Windows/CPU/x86-64_vs2017_avx_openblas/paddle_inference.zip"
) else if "%1" == "cu110" (
  set DOWNLOAD_URL="https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.0_cudnn8/paddle_inference.zip"
)

if exist %FILEPATH% (
    echo Found %FILEPATH%
) else (
    echo Couldn't find %FILEPATH%, downloading it ...
    echo Downloading from: %DOWNLOAD_URL%
    powershell -Command "(New-Object Net.WebClient).DownloadFile('%DOWNLOAD_URL%', '%cd%\paddle.zip')"
    powershell -Command "Expand-Archive -LiteralPath paddle.zip -DestinationPath %cd%\paddle"
    rename paddle_inference_install_dir paddle
    del /f paddle.zip
    echo Finished downloading paddle
)

if exist build rd /q /s build
md build\classes
cd build
javac -sourcepath ../../paddlepaddle-engine/src/main/java/ ../../paddlepaddle-engine/src/main/java/ai/djl/paddlepaddle/jni/PaddleLibrary.java -h include -d classes
cmake .. -A x64
cmake --build . --config Release
