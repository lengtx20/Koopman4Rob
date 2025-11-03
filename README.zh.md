## 安装

```bash
conda create -n koopman4rob python=3.10
conda activate koopman4rob
pip3 install -r requirements.txt
```


## 配置

在 `configs/config.yaml` 中配置各种参数，全部可配置参数请参考 `config.py`中的各种配置类（符合hydra规范）。

### 数据集加载

默认配置中通过`configs/data.py` 加载数据集，可以修改该文件以加载自定义数据集。
数据集`datasets`是一个列表，结构为[dataset1， dataset2， ...]，每个dataset是包含多个episode
可迭代对象的列表，即`datasets = [episode1, episode2, ...]`。其中每个episode是一个Iterable[Dict[str, np.ndarray]]类型的对象。其中，多个dataset不是串行关系，而是会被并行加载并将字典数据合并在一起。例如：

```python
dataset1_episode1 = [{'robot/state': np.array([...])}, ...]
dataset2_episode1 = [{'camera/image': np.array([...])}, ...]
dataset1 = [dataset1_episode1, ...]
dataset2 = [dataset2_episode1, ...]
datasets = [dataset1, dataset2]
```
则在加载后，每个数据字典将包含`'robot/state'`和`'camera/image'`两个key。其中key分为state和action两类，分别对应系统状态和控制输入，可以参考`configs/data.py`中对`DataLoaderConfig`的配置。每个类中可以指定多个key，加载时会按顺序将对应的数组拼接在一起作为模型最终的状态和动作输入。

### 模型配置

模型的可配置参数位于配置文件中的`model`字段，其中`state_dim`和`action_dim`分别对应系统状态和控制输入的维度，默认通过数据进行推断，故省略配置。


### 训练配置

训练相关的配置位于配置文件中的`train`字段，详细说明可参考`config.py`中`TrainConfig`类的注释。


## 训练

```bash
python3 main_blip2.py +mode=train
```

可以通过命令行参数覆盖配置文件中的参数（符合hydra规范），例如修改`batch_size`：

```bash
python3 main_blip2.py +mode=train +data_loader.batch_size=128
```

训练可以随时通过`Ctrl+C`中断，模型检查点会自动保存到`logs/checkpoints`目录下。

## 测试

```bash
python3 main_blip2.py +mode=test +checkpoint_path=14/best
```
其中`checkpoint_path`指定要加载的模型检查点路径，相对于`logs/checkpoints`目录。
请注意修改配置文件中的数据集配置以加载测试数据集。


## 推理

```bash
python3 main_blip2.py +mode=infer +checkpoint_path=14/best
```
