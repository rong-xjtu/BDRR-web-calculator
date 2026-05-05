# 医学机器学习模型 - Gradient Boosting

## 模型信息

- 模型类型: Gradient Boosting
- 训练时间: 20260121_161844
- 测试准确率: 0.8155
- 测试AUC: 0.8533
- 交叉验证: 0.7970 ± 0.0364

## 特征信息

- 特征数量: 10
- 选择的特征: Age, Histologic grade, ER proportion, PR proportion, Ki-67 proportion, p53, Preoperative Long /short diameter, PR status, HER-2 IHC, Family history

## 文件说明

- `best_model.pkl`: 训练好的最佳模型
- `preprocessing_pipeline.pkl`: 数据预处理管道
- `model_metadata.json`: 模型详细信息和性能指标
- `feature_importance.csv`: 特征重要性排序
- `data_statistics.json`: 训练数据统计信息

## 使用方法

```python
from model_inference import ModelInference

# 加载模型
inference = ModelInference("saved_models\best_model_20260121_161844")

# 对新数据进行预测
predictions = inference.predict(new_data)
probabilities = inference.predict_proba(new_data)
```
