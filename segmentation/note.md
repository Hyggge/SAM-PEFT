# 代码阅读

## 命令行解析

### test.py

这段代码是一个命令行参数解析的函数。它使用argparse模块创建一个ArgumentParser对象parser，并定义了一系列的命令行参数选项。

这些命令行参数选项包括：

- 'config': 用于指定配置文件路径
- 'checkpoint': 用于指定checkpoint文件路径
- '--work-dir': 用于指定评估指标结果的保存目录
- '--aug-test': 用于指定是否使用Flip和Multi scale aug进行测试
- '--out': 用于指定输出结果文件的pickle格式
- '--format-only': 仅格式化输出结果而不执行评估
- '--eval': 用于指定评估指标，例如'mIoU'和'cityscapes'
- '--show': 用于指定是否显示结果
- '--show-dir': 用于指定保存可视化结果的目录
- '--gpu-collect': 是否使用GPU进行结果收集
- '--tmpdir': 用于指定结果收集的临时目录
- '--options': 用于指定覆盖配置文件中的一些设置，键值对格式为xxx=yyy，支持嵌套list/tuple值
- '--cfg-options': 用于覆盖配置文件中的一些设置，键值对格式为xxx=yyy，支持嵌套list/tuple值
- '--eval-options': 用于自定义评估选项
- '--launcher': 用于指定任务调度器类型
- '--opacity': 用于指定绘制分割图的透明度
- '--local_rank': 用于指定本地rank值
- 
最后，使用parser.parse_args()解析命令行参数，并返回args对象。在解析之前，代码还会做一些验证和兼容性处理，例如判断是否同时指定了args.options和args.cfg_options，以及向用户发出警告。如果环境变量中没有设置LOCAL_RANK，代码会将args.local_rank的值赋给LOCAL_RANK环境变量。最后返回解析后的args对象。


## MMSsg项目的组成部件
- 数据集和数据加载器
  - 数据集包含数据的读取和预处理pipeline
- 模型
  - 模型本身包含损失函数（loss）
- 优化器
  - 优化器一般被封装到optimizer_wrapper中
- 评估指标



## 自定义一个部件
- 可以直接通过import引进项目中
- 可以通过配置文件中的custom_imports


## Other
参数调度器：lr_config(0.x), param_scheduler(latest)
不同层设置不同的超参数 (0.x / latest)

处理二值分割任务: https://mmsegmentation.readthedocs.io/zh_CN/0.x/faq.html#id2
模型库(backbone/algorithm): https://mmsegmentation.readthedocs.io/zh_CN/0.x/modelzoo_statistics.html


Swin, PVT, 和 VIT 都是backbone,
- 这三者之间的关系？
- 但是论文里提到的settings of PVT， settings of Swin 是什么意思？

semantic FPN, UpperNet 是算法框架
- 开源代码中只有mask2former 但是没有 semantic FPN，什么是mask2former？和 semantic FPN 有什么关系

DeiT、AugReg、BEiT 都是预训练好的参数？论文里有提到


