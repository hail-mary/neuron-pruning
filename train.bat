@echo off
setlocal enabledelayedexpansion

REM List of environments
set ENVS=HalfCheetah-v5 Swimmer-v5 Walker2d-v5

for %%E in (%ENVS%) do (
    echo.
    echo ========================================
    echo Processing Environment: %%E
    echo ========================================

    REM 1. Proposed method (Seeds 1 to 10)
    echo Running Proposed method for %%E...
    for /L %%S in (1,1,10) do (
        echo   - Seed %%S
        python main.py --logdir data/%%E/proposed --env %%E --seed %%S
    )

    REM 2. PPO-WA method (Seeds 1 to 10, update_interval > num_iterations)
    echo Running PPO-WA method for %%E...
    for /L %%S in (1,1,10) do (
        echo   - Seed %%S
        python main.py --logdir data/%%E/PPO-WA --env %%E --seed %%S --update_interval 1001
    )
)

echo.
echo All training tasks for HalfCheetah-v5, Swimmer-v5, and Walker2d-v5 (10 seeds each) are completed.
pause