# Plan of Holy Question Answering
## Reference

## RoadMap
- 20180725 solution_keras 调试通过单个index, 但是精度太低
-

## Problems
1. loss 太小, 使用tensorboard 查看各层各个变量的输出情况
    + 解决 : myLayer 中 使用 batch_dot, 结果中得到正常loss
2. 单个index 预测准确率太低
3. 单个index 预测准确率有提升后, 加入第二个 index 预测
4. 联合考虑两个index, 对最终答案进行预测
5. 参照论文方案实现, perdict的结果
