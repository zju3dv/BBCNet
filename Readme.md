## training

```
srun -p 3d_share -n4 --cpus-per-task=5 --gres=gpu:4 --ntasks-per-node=4 -N1 bash train.sh

srun -p 3d_share -n4 --cpus-per-task=5 --gres=gpu:4 -w SH-IDC1-10-198-8-153 --ntasks-per-node=4 -N1 bash train.sh

```

