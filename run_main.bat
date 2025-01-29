@echo off

REM Execute the Python script with the specified arguments
python main.py --logdir drop_ant --env Ant-v5
python main.py --logdir drop_cheetah --env HalfCheetah-v5
python main.py --logdir drop_hopper --env Hopper-v5
python main.py --logdir drop_pusher --env Pusher-v5
python main.py --logdir drop_reacher --env Reacher-v5
python main.py --logdir drop_walker --env Walker2d-v5

REM Print a message
echo Script execution completed.