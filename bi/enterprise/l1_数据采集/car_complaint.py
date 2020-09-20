# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 汽车投诉信息采集：
数据源：http://www.12365auto.com/zlts/0-0-0-0-0-0_0-0-1.shtml
投诉编号，投诉品牌，投诉车系，投诉车型，问题简述，典型问题，投诉时间，投诉状态
"""
import os

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 '
                  'Safari/537.36 '
}

CHE_ZHI_BASE_URL = 'http://www.12365auto.com/zlts/0-0-0-0-0-0_0-0-0-0-0-0-0-{}.shtml'


def get_page_content(url):
    """
    获取页面内容

    :param url:
    :return:
    """
    html = requests.get(url, headers=HEADERS, timeout=10)
    content = html.text
    soup = BeautifulSoup(content, features='html5lib')
    return soup


def extract_content(soup):
    """
    根据 bs 对象提取页面内容

    :param soup:
    :return:
    """
    # 投诉编号，投诉品牌，投诉车系，投诉车型，问题简述，典型问题，投诉时间，投诉状态
    df = pd.DataFrame(columns=['complaint_id', 'complaint_brand', 'complaint_model', 'complaint_type', 'complaint_desc',
                               'complaint_problem', 'complaint_time', 'complaint_status'])

    # 找到完整的投诉框
    tslb_b = soup.find('div', class_='tslb_b')

    for tr in tslb_b.find_all('tr'):
        tmp = dict()
        td_list = tr.find_all('td')
        # 第一个 tr 没有 td，其余都有 8 个 td
        if len(td_list) > 0:
            tmp['complaint_id'] = td_list[0].text
            tmp['complaint_brand'] = td_list[1].text
            tmp['complaint_model'] = td_list[2].text
            tmp['complaint_type'] = td_list[3].text
            tmp['complaint_desc'] = td_list[4].text
            tmp['complaint_problem'] = td_list[5].text
            tmp['complaint_time'] = td_list[6].text
            tmp['complaint_status'] = td_list[7].text
            df = df.append(tmp, ignore_index=True)

    return df


def get_complaint_data(path):
    """
    获取投诉数据

    :param path:
    :return:
    """
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        page_num = 20
        request_base_url = CHE_ZHI_BASE_URL
        res_df = pd.DataFrame(
            columns=['complaint_id', 'complaint_brand', 'complaint_model', 'complaint_type', 'complaint_desc',
                     'complaint_problem', 'complaint_time', 'complaint_status'])
        for i in range(page_num):
            request_url = request_base_url.format(i + 1)
            soup = get_page_content(request_url)
            extract_df = extract_content(soup)
            res_df = res_df.append(extract_df)

        res_df.to_csv(path, index=False)
        return res_df


def main():
    path = './data/car_complaint.csv'
    df = get_complaint_data(path)
    print(df)


if __name__ == '__main__':
    main()
