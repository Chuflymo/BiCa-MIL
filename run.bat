set VIRTUALENV_PATH=C:\Users\95188\.conda\envs\torch2.3.0
set PYTHONPATH=%VIRTUALENV_PATH%\Lib;%VIRTUALENV_PATH%\Lib\site-packages
set PYTHON_EXE=%VIRTUALENV_PATH%\python.exe

call %PYTHON_EXE% main.py ^
    --datasets=camelyon16 ^
    --dataset_root="D:\DATA\PLIP_MIL\RAW_PLIP_features" ^
    --fewshot_samples=-1 ^
    --seed=2024 ^
    --n_classes=2 ^
    --epochs=100 ^
    --patients=30 ^
    --lr=1e-4 ^
    --model=adapter_mil ^
    --init_alpha=1 ^
    --init_beta=1 ^
    --plip_model_path="D:\WORK\Projects\PLIP_test\plip_finish" ^
    --save_path="D:\DATA\PLIP_MIL\plip_adapter_mil_model"

pause
pause