## 安装

```bash
conda create -n koopman4rob python=3.10
conda activate koopman4rob
pip3 install -r requirements.txt
```

## 配置

在 `configs/config.yaml` 中配置各种参数，全部可配置参数请参考 `config.py`中的各种配置类（符合hydra规范）。

### 数据集加载

数据集默认配置位于`configs/dataset/train.yaml` ，可以修改该文件以加载自定义数据集。
数据集`datasets`是一个可迭代对象，结构为[dataset1， dataset2， ...]，每个dataset是包含多个episode
可迭代对象的列表，即`datasets = [episode1, episode2, ...]`。其中每个episode是一个Iterable[Dict[str, np.ndarray]]类型的对象。其中，多个dataset不是串行关系，而是会被并行加载并将字典数据合并在一起。例如：

```python
dataset1_episode1 = [{'robot/state': np.array([...])}, ...]
dataset2_episode1 = [{'camera/image': np.array([...])}, ...]
dataset1 = [dataset1_episode1, ...]
dataset2 = [dataset2_episode1, ...]
datasets = [dataset1, dataset2]
```
则在加载后，每个数据字典将包含`'robot/state'`和`'camera/image'`两个key。

### 模型配置

模型的默认配置位于`configs/model/single.yaml`文件，其中使用了自定义的`DeepKoopman`，全部配置项参考`DeepKoopmanConfig`配置类，其中`state_dim`和`action_dim`分别对应系统状态和控制输入的维度，默认通过数据进行推断，故省略配置。通常，不同的模型结构会对`DataLoader`有相应的要求，因此，在模型配置文件中可以额外增加`data_loader`字段以覆盖默认的主配置。其中`stack`字段用于对数据进行按行、列堆叠处理，通常用于对不同种类的数据进行拼接，以及构建时间序列。


### 训练配置

默认配置位于`config.yaml`中的`train`字段，详细说明可参考`config.py`中`TrainConfig`类的注释。
可主要关注`train_val_split`字段正确划分训练集和验证集。

### 推理配置

模型推理可分为：

- 数据集推理（单步）：使用数据集中的数据进行推理，将模型预测的动作与数据集中的真实动作进行对比，评估模型性能（与test阶段类似）。可选择是否将预测的动作通过具体的软硬件平台接口执行。
- 平台（sim or real）推理（链式）：使用数据集中的或给定的初始状态或动作对具体平台初始化，然后通过平台接口获取后续的状态反馈，将模型预测的动作通过相关接口执行，评估模型在该平台上的性能。
  - 开环：将模型的预测的动作作为模型的下一个状态输入（除了平台初始化）。适用于动作和状态物理意义相同的情况。
  - 闭环：将平台反馈的状态作为模型的下一个状态输入。
- 单/多模型推理：推理时可以使用单一模型或将多个模型集成（拼接、综合等）使用。

各种推理的配置文件位于`configs/infer`目录下，默认使用`test_single.yaml`进行数据集推理。推理的通用配置位于`config.yaml`中的`infer`字段，详细说明可参考`config.py`中`InferConfig`类的注释。此外，为了支持各种平台和不同的推理方式，额外增加了一个`interactor`字段，用于配置具体的定制化的交互器，默认类位于`configs.inter.Interactor`，可供参考。


### 配置保存

目前，模型训练结束后会将完整配置一并保存到检查点根目录中（如`logs/checkpoints/0/training_config.yaml`），其中模型配置会额外保存到每个检查点目录，文件名为`config.yaml`。对于支持自定义类型的实例参数（使用`_target_`动态加载），例如`datasets`，保存参数时会尝试调用其`dump`方法，该方法需要自行实现以返回任何普通Python数据类型（如dict、list、str等），以便进行JSON序列化。如果找不到`dump`方法或方法出现异常，则会尝试获取`config`属性，该属性类型可以是pydantic BaseModel，dataclass和dict之一。若上述方式均不满足，最终会使用`repr`进行保存。

### 命令行参数

可以通过命令行参数覆盖配置文件中的参数（符合hydra规范）。

## 训练

```bash
python3 main.py
```

命令行覆盖示例（修改`batch_size`）：

```bash
python3 main.py data_loader.batch_size=128
```

训练可以随时通过`CTRL+C`中断，模型检查点会自动保存到`logs/checkpoints/<ID>`目录下，默认会产生`best`，`last`，以及若干位于`val_loss`目录下以实际验证损失命名的检查点。

## 测试

```bash
python3 main.py stage=test +checkpoint_path=0/best
```
其中`checkpoint_path`指定要加载的模型检查点路径，相对于`logs/checkpoints`目录。
请注意修改配置文件中的数据集配置以加载测试数据集。


## 推理

以默认配置举例：

### 数据集推理

- 不执行动作
```bash
python3 main.py +infer=basis +checkpoint_path=0/best
```

- 执行来自数据集的动作（类似数据重放）
```bash
python3 main.py +infer=basis interactor.action_from=data_loader +checkpoint_path=0/best infer.frequency=0 infer.rollout_wait=input
```

- 执行来自模型预测的动作
```bash
python3 main.py +infer=basis interactor.action_from=model +checkpoint_path=0/best infer.frequency=0 infer.rollout_wait=input
```

### 平台推理

- 开环
```bash
python3 main.py +infer=real +checkpoint_path=0/best
```

- 闭环
```bash
python3 main.py +infer=real interactor.open_loop_predict=false +checkpoint_path=0/best
```
