@if "%4" == "" (
    @set MS_ROOT=%VULCAN_INSTALL_DIR%
) else (
    @set MS_ROOT=%4
)

@echo Setting Environment
@set PATH=%MS_ROOT%\windsdk\7.1\bin;%MS_ROOT%\msvc100sp1\VC\bin\amd64;%MS_ROOT%\windsdk\7.1\bin\NETFX 4.0 Tools\x64;%MS_ROOT%\windsdk\7.1\bin\x64;%PATH%
@set INCLUDE=%MS_ROOT%\msvc100sp1\VC\include\;%MS_ROOT%\windsdk\7.1\include\;%INCLUDE%
@set LIB=%MS_ROOT%\msvc100sp1\VC\bin\amd64\;%MS_ROOT%\windsdk\7.1\Lib\x64\;%MS_ROOT%\msvc100sp1\VC\lib\amd64\;%LIB%

@echo Printing Environment
@echo PATH=%PATH%
@echo INCLUDE=%INCLUDE%
@echo LIB=%LIB%

@echo Rebuilding %1 %2 %3
%MS_ROOT%\dotNet\4.0\Framework64\v4.0.30319\MSBUILD.exe /t:rebuild %1 /p:Platform=%2 /p:Configuration=%3 /p:CudaToolkitDir=%MS_ROOT%\cuda\ /p:VCTargetsPath=%MS_ROOT%\dotNet\4.0\MSBuild\Microsoft.Cpp\v4.0\ /p:VCTargetsPath10=%MS_ROOT%\dotNet\4.0\MSBuild\Microsoft.Cpp\v4.0\ /p:FrameworkPathOverride=%MS_ROOT%\dotNet\4.0\Reference_assemblies\v4.0\ /p:MSBuildOverrideTasksPath=%MS_ROOT%\dotNet\4.0\Framework64\v4.0.30319 /p:VCInstallDir=%MS_ROOT%\msvc100sp1\VC\ /p:WindowsSdkDir=%MS_ROOT%\windsdk\7.1\ /p:UseEnv=true /p:TrackFileAccess=false /p:CUDA_PRODUCT_VERSION=%VULCAN_CUDA_VERSION%
