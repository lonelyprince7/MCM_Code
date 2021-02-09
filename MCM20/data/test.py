'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-31 01:39:04
LastEditors: ZhangHongYu
LastEditTime: 2021-01-31 02:03:18
'''


def foo(n):
    return lambda i:  i+n


test = foo(2)
print(test(1))
