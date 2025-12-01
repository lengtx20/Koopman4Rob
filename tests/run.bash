ckpt_id=6
com="data_loaders.pairwise.gap=10"
task_name="data_loaders.task_name=reach_block"
data_cfg="${com} ${task_name}"
# mode="+infer=act ${data_cfg}"
# mode="+infer=act ${data_cfg} interactor.action_from=model interactor.extractor.enable=true"
mode="+infer=real ${data_cfg}"
# mode="+infer=real_only"
python3 main.py ${mode} +checkpoint_path=${ckpt_id}/best
# python3 main.py +train_val=basis ${task_name} ${com}
