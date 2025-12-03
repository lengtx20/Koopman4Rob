
set -ex

ckpt_id=6
# ckpt_path=${ckpt_id}/best
ckpt_path=multi/0.yaml
com="+data_loaders.pairwise.gap=10"
# com="+data_loaders.pairwise.gap=0,1,2,3,4,5,6,7,8,9"
task_name="data_loaders.task_name=reach_block"
data_cfg="${com} ${task_name}"
# model_cfg="model=single"
model_cfg="model=multi"
# mode="+infer=act ${data_cfg} ${model_cfg}"
# mode="+infer=act_mock ${data_cfg} ${model_cfg}"
# mode="+infer=act ${data_cfg} ${model_cfg} interactor.action_from=model interactor.extractor.enable=true +infer.start_rollout=60"
mode="+infer=real ${data_cfg} ${model_cfg}"
# mode="+infer=real_only ${model_cfg}"
python3 main.py ${mode} +checkpoint_path=${ckpt_path}
# python3 main.py -m +train_val=basis ${data_cfg}

# python3 main.py +infer=static +checkpoint_path=5/best data_loaders.task_name=reach_center_new data_loaders.pairwise.gap=10
