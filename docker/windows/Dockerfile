ARG version=ltsc2019
FROM mcr.microsoft.com/windows/servercore:$version

ADD https://aka.ms/vs/17/release/vc_redist.x64.exe /vc_redist.x64.exe
RUN C:\vc_redist.x64.exe /quiet /install
RUN del C:\vc_redist.x64.exe

ENV chocolateyUseWindowsCompression false

RUN powershell -Command \
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1')); \
    choco feature disable --name showDownloadProgress

RUN choco install -y openjdk17
