
set -ex

ckpt_id=6
com="data_loaders.pairwise.gap=0,1,2,3,4,5,6,7,8,7,8,9"
task_name="data_loaders.task_name=reach_center_new"
data_cfg="${com} ${task_name}"
# mode="+infer=act ${data_cfg}"
# mode="+infer=act ${data_cfg} interactor.action_from=model interactor.extractor.enable=true"
mode="+infer=real ${data_cfg}"
# mode="+infer=real_only"
# python3 main.py ${mode} +checkpoint_path=${ckpt_id}/best
python3 main.py -m +train_val=basis ${data_cfg} hydra/launcher=basic

# python3 main.py +infer=static +checkpoint_path=5/best data_loaders.task_name=reach_center_new data_loaders.pairwise.gap=10
