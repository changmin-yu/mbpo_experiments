export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/v-changminyu/.mujoco/mjpro150/bin

python main_mbpo_dm_control.py --seed 123 --rollout_max_epoch 150 --num_epoch 1000 --num_train_repeat 1 --domain_name walker --task_name walk & \
python main_mbpo_dm_control.py --seed 123 --rollout_max_epoch 150 --num_epoch 1000 --num_train_repeat 1 --domain_name hopper --task_name hop & \
python main_mbpo_dm_control.py --seed 123 --rollout_max_epoch 150 --num_epoch 1000 --num_train_repeat 1 --domain_name finger --task_name spin & \
python main_mbpo_dm_control.py --seed 123 --rollout_max_epoch 150 --num_epoch 1000 --num_train_repeat 1 --domain_name walker --task_name run & \
python main_mbpo_dm_control.py --seed 123 --rollout_max_epoch 150 --num_epoch 1000 --num_train_repeat 1 --domain_name hopper --task_name stand & \
