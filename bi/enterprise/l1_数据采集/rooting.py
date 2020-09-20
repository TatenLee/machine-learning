# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 
"""

"""
0   10
0   5
2   5
"""


def find_critical_num(target_num, power):
    """
    找到目标数字的开方临界数字
    :param target_num:
    :param power:
    :return:
    """
    low = 0
    high = target_num
    while low < high:
        mid = (low + high) / 2
        if mid ** power <= target_num <= (mid + 1) ** power:
            return mid
        if target_num < mid ** power:
            high = mid
        if target_num > mid ** power:
            low = mid


def rooting(target_num, power, place):
    """
    求数字的指定次开方

    :param target_num:
    :param power:
    :param place:
    :return:
    """
    critical_num = find_critical_num(target_num, power)
    print(critical_num)

    low = critical_num
    high = critical_num + 1
    mid = (low + high) / 2

    while high - low > place:
        if target_num < mid ** power:
            high = mid
        if target_num > mid ** power:
            low = mid
        mid = (low + high) / 2

    return mid


if __name__ == '__main__':
    num = 10
    root = 2
    decimal_place = 1e-10
    res = rooting(num, root, decimal_place)
    print(res)
