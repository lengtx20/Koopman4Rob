ckpt_id=0
com="data_loaders.pairwise.gap=10"
task_name="data_loaders.task_name=reach_center_1130"
data_cfg="${com} ${task_name}"
# mode="+infer=act ${data_cfg}"
mode="+infer=real ${data_cfg}"
# mode="+infer=real_only"
python3 main.py ${mode} +checkpoint_path=${ckpt_id}/best
# python3 main.py +train_val=basis ${task_name} ${com}
