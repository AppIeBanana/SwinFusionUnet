import tensorflow as tf
from tensorflow.python.summary import summary_iterator

# 读取Tensorboard文件并解析事件数据
for event in summary_iterator('tf-logs'):
    for value in event.summary.value:
        print(value.tag, value.simple_value)
