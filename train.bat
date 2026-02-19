@echo off

REM Execute the Python script with the specified arguments
python main.py --logdir ant --env Ant-v5
@REM python main.py --logdir cheetah --env HalfCheetah-v5
@REM python main.py --logdir hopper --env Hopper-v5
@REM python main.py --logdir pusher --env Pusher-v5
@REM python main.py --logdir reacher --env Reacher-v5
@REM python main.py --logdir swimmer --env Swimmer-v5
@REM python main.py --logdir walker --env Walker2d-v5

REM Print a message
echo Script execution completed.