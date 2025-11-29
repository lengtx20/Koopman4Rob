com="data_loaders.pairwise.gap=10"
ckpt_id=5
task_name="data_loaders.task_name=reach_center_new"
compare="interactor.extractor.enable=true interactor.model_input_from=data_loader +interactor.save_image=true"
action="interactor.action_from=model"
python3 main.py ${com} ${task_name} ${compare} ${action} +infer=act +checkpoint_path=${ckpt_id}/best
