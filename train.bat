@echo off
setlocal enabledelayedexpansion

REM List of environments
set ENVS=Ant-v5 HalfCheetah-v5 Swimmer-v5 Walker2d-v5

for %%E in (%ENVS%) do (
    echo.
    echo ========================================
    echo Processing Environment: %%E
    echo ========================================

    REM 1. Proposed method (10 seeds)
    echo Running Proposed method for %%E...
    for /L %%S in (1,1,10) do (
        echo   - Seed %%S
        python main.py --logdir data_reproduced/%%E/proposed --env %%E --seed %%S
    )

    REM 2. PPO-WA method (10 seeds, no pruning)
    echo Running PPO-WA method for %%E...
    for /L %%S in (1,1,10) do (
        echo   - Seed %%S
        python main.py --logdir data_reproduced/%%E/PPO-WA --env %%E --seed %%S --update_interval 1001
    )

    REM 3. Structured-GMP method (10 seeds, pruning only)
    echo Running Structured-GMP method for %%E...
    for /L %%S in (1,1,10) do (
        echo   - Seed %%S
        python baseline_train.py --logdir data_reproduced/%%E/Structured-GMP --env %%E --seed %%S
    )
)

echo.
echo All training tasks for MuJoCo environments (10 seeds each) are completed.
pause