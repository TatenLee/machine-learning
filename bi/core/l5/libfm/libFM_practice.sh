#!/bin/bash

# 格式化训练集
perl triple_format_to_libfm.pl -in ./titanic/train.csv -target 1 -delete_column 0 -separator ","

# 格式化测试集
perl triple_format_to_libfm.pl -in ./titanic/test.csv -target 1 -delete_column 0 -separator ","

# 训练,指定优化方式为SGD
./libfm-1.42.src/bin/libFM -task c -method sgd -learn_rate 0.01 -train ./titanic/train.csv.libfm -test ./titanic/test.csv.libfm -dim '1,1,8' -out ./titanic_out.txt
