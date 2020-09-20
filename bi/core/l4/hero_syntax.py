# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 随机组合语法
"""

import random

# 语法
grammar = '''
战斗 => 施法 结果
施法 => 主语 动作 技能 
结果 => 主语 获得 效果
主语 => 张飞 | 关羽 | 赵云 | 典韦 | 许褚 | 刘备 | 黄忠 | 曹操 | 鲁班七号 | 貂蝉
动作 => 施放 | 使用 | 召唤 
技能 => 一骑当千 | 单刀赴会 | 青龙偃月 | 刀锋铁骑 | 黑暗潜能 | 画地为牢 | 守护机关 | 狂兽血性 | 龙鸣 | 惊雷之龙 | 破云之龙 | 天翔之龙
获得 => 损失 | 获得 
效果 => 数值 状态
数值 => 1 | 1000 |5000 | 100 
状态 => 法力 | 生命
'''


def get_grammar_dict(gram, line_sep='\n', grammar_sep='=>'):
    grammar_dict = dict()
    for line in gram.split(line_sep):
        if not line.strip():
            continue
        else:
            expr, statement = line.split(grammar_sep)
            grammar_dict[expr.strip()] = [i.split() for i in statement.split('|')]
    return grammar_dict


def generate(gram_dict, target, is_eng=False):
    if target not in gram_dict:
        return target
    else:
        find = random.choice(gram_dict[target])
        blank = ''
        # 如果是英文，中间间隔为空格
        if is_eng:
            blank = ' '
        return blank.join(generate(gram_dict, t, is_eng) for t in find)


grammar_dict = get_grammar_dict(grammar)
print(grammar_dict)
print(generate(grammar_dict, '战斗'))
print(generate(grammar_dict, '战斗', True))
