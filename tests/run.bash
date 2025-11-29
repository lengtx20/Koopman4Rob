com="data_loaders.pairwise.gap=10"
ckpt_id=5
task_name="data_loaders.task_name=reach_center_new"
compare="interactor.extractor.enable=true interactor.model_input_from=env +interactor.save_image=true"

python3 main.py ${com} ${task_name} ${compare} +infer=act +checkpoint_path=${ckpt_id}/best
