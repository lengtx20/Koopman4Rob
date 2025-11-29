com="data_loaders.pairwise.gap=10"
ckpt_id=5
task_name=data_loaders.task_name=reach_center_new

python3 main.py ${com} +infer=act +checkpoint_path=${ckpt_id}/best
