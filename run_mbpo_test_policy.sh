export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/v-changminyu/.mujoco/mjpro150/bin

python mbpo_test_policy.py --seed 123456 --rollout_max_epoch 100 --num_epoch 125