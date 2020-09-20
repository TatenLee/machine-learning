# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lishuang
@description: 基于高德地图的路径规划
从指定地点start，到终点end的路径规划
最优路径定义：
1）距离最短
2）时间最短
输入：start,end
输出：路径规划，所需的距离、时间
"""

import logging
import os
import pickle
import re
from collections import defaultdict

import pandas as pd
import requests
from bs4 import BeautifulSoup

LOGGER_FORMATTER = '[%(asctime)s]-[%(filename)s:%(lineno)s]-[%(levelname)s]- %(message)s'

HEADER = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/74.0.3729.131 Safari/537.36'
}

AMAP_WEB_KEY = 'f9fcab9665efff83bca5214342896e23'

BEIJING_LINE_REQUEST_URL = 'https://ditie.mapbar.com/beijing_line/'

CITY = '北京'

logging.basicConfig(format=LOGGER_FORMATTER)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def get_content_soup(url):
    """
    通过请求获取页面信息

    :param url:
    :return:
    """
    html = requests.get(url, headers=HEADER, timeout=10)
    content = html.text
    soup = BeautifulSoup(content, 'html.parser')
    return soup


def parse_subway_data(soup, city):
    """
    将爬取的数据进行处理

    :param soup:
    :param city:
    :return:
    """
    data = pd.DataFrame(columns=['name', 'site'])
    stations = soup.find_all('div', class_='station')
    for station in stations:
        line = station.find('strong', class_='bolder').text
        station_names = station.find('ul').find_all('a')
        for station_name in station_names:
            data = data.append({'name': station_name.text, 'site': line}, ignore_index=True)
    data['city'] = city
    logger.debug(data)
    return data


def create_or_load_subway_data(subway_data_path):
    """
    创建或加载数据

    :param subway_data_path:
    :return:
    """
    if os.path.exists(subway_data_path):
        data = pd.read_csv(subway_data_path)
    else:
        soup = get_content_soup(BEIJING_LINE_REQUEST_URL)
        data = parse_subway_data(soup, CITY)
        data.to_csv(subway_data_path, encoding='utf-8', index=False)
    return data


def create_or_load_subway_location_data(subway_data_path, subway_location_data_path):
    """
    创建或加载数据

    :param subway_data_path:
    :param subway_location_data_path:
    :return:
    """
    if os.path.exists(subway_location_data_path):
        data = pd.read_csv(subway_location_data_path)
    else:
        data = create_or_load_subway_data(subway_data_path)
        data['longitude'], data['latitude'] = None, None
        for idx, row in data.iterrows():
            name = row['name']
            city = row['city']
            longitude, latitude = get_location(name, city)
            data.iloc[idx]['longitude'] = longitude
            data.iloc[idx]['latitude'] = latitude
        data.to_csv(subway_location_data_path, encoding='utf-8', index=False)
    return data


def get_location(keywords, city):
    """
    根据关键词获取位置

    :param keywords: 
    :param city:
    :return: 
    """
    url = f'http://restapi.amap.com/v3/place/text?key={AMAP_WEB_KEY}&keywords={keywords}' \
          f'&types=&city={city}&children=1&offset=1&page=1&extensions=all'
    data = requests.get(url, headers=HEADER, timeout=10)
    data.encoding = 'utf-8'
    data = data.text
    logger.debug(f'keywords: {keywords}, city: {city}, data: {data}')
    pattern = 'location":"(.*?),(.*?)"'
    try:
        result = re.findall(pattern, data)
        return result[0][0], result[0][1]
    except Exception as e:
        logger.error(f'正常解析错误: {e}, data: {data}')
        return get_location(keywords.replace('站', ''), city)


def compute_distance(longitude1, latitude1, longitude2, latitude2):
    """
    获取两点之间的距离和时间

    :param longitude1:
    :param latitude1:
    :param longitude2:
    :param latitude2:
    :return: distance, duration
    """
    url = f'http://restapi.amap.com/v3/distance?key={AMAP_WEB_KEY}&origins={longitude1},{latitude1}' \
          f'&destination={longitude2},{latitude2}&type=1'
    data = requests.get(url, headers=HEADER, timeout=10)
    data.encoding = 'utf-8'
    data = data.text
    pattern = '"distance":"(.*?)","duration":"(.*?)"'
    result = re.findall(pattern, data)
    return result[0][0], result[0][1]


def create_or_load_distance_graph(df, distance_data_path):
    """
    创建或加载站点距离图

    :param df:
    :param distance_data_path:
    :return:
    """
    if os.path.exists(distance_data_path):
        file = open(distance_data_path, 'rb')
        return pickle.load(file)
    else:
        graph = defaultdict(dict)
        for i in range(df.shape[0] - 1):
            site1 = df.iloc[i]['site']
            site2 = df.iloc[i + 1]['site']
            if site1 == site2:
                longitude1, latitude1 = df.iloc[i]['longitude'], df.iloc[i]['latitude']
                longitude2, latitude2 = df.iloc[i + 1]['longitude'], df.iloc[i + 1]['latitude']
                name1, name2 = df.iloc[i]['name'], df.iloc[i + 1]['name']
                distance, duration = compute_distance(longitude1, latitude1, longitude2, latitude2)
                graph[name1][name2] = distance
                graph[name2][name1] = distance
                output = open(distance_data_path, 'wb')
                pickle.dump(graph, output)
        return graph


def find_lowest_cost_node(costs, processed_node):
    """
    找到花销最低的节点

    :param costs:
    :param processed_node:
    :return:
    """
    lowest_cost = float('inf')
    lowest_cost_node = None
    for node in costs:
        if node not in processed_node:
            # 如果当前节点的 cost 比已经存在的 cost 小，那么擂主更新，即更新这个节点为 cost 最小的节点
            if costs[node] < lowest_cost:
                lowest_cost = costs[node]
                lowest_cost_node = node
    return lowest_cost_node


def find_shortest_path(start, end, parents):
    """
    找到最短路径

    :param start:
    :param end:
    :param parents:
    :return:
    """
    node = end
    shortest_path = [end]
    while parents[node] != start:
        shortest_path.append(parents[node])
        node = parents[node]
    shortest_path.append(start)
    return shortest_path


def get_stations_costs_and_parents(graph, start, end):
    """
    获取站点间的花销和节点

    :param graph:
    :param start:
    :param end:
    :return:
    """
    costs = dict()
    parents = dict()
    # 获取节点的邻居节点
    for node in graph[start].keys():
        costs[node] = float(graph[start][node])  # 距离
        parents[node] = start  # 连接的节点
    costs[end] = float('inf')
    return costs, parents


def dijkstra(graph, costs, parents, start, end):
    """
    使用 dijkstra 计算最短距离

    :param graph:
    :param costs:
    :param parents:
    :param start:
    :param end:
    :return:
    """
    processed_node = list()
    node = find_lowest_cost_node(costs, processed_node)
    logger.info(f'当前 cost 最小节点为: {node}')
    # 只要有 cost 最小的节点，就进行路径计算，如果所有节点都在 processed 中即结束
    while node is not None:
        # 获取节点目前的 cost
        cost = costs[node]
        # 获取节点的邻居节点
        neighbors = graph[node]
        # 遍历所有邻居，看是否可以通过 node，完成 cost 更新
        for neighbor in neighbors.keys():
            # 计算经过当前节点达到邻居节点的 cost
            new_cost = cost + float(neighbors[neighbor])
            # 如果通过 node，可以更新 start -> neighbor 的 cost
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                parents[neighbor] = node
        # 将当前节点标记为处理过
        processed_node.append(node)
        # 找出接下来要处理的节点并循环
        node = find_lowest_cost_node(costs, processed_node)

    # 循环完毕说明所有节点都处理完毕
    shortest_path = find_shortest_path(start, end, parents)
    shortest_path.reverse()
    return shortest_path


def run():
    beijing_subway_data_path = 'data/beijing_subway.csv'
    beijing_subway_location_data_path = 'data/beijing_subway_location.csv'
    beijing_subway_site_distance_graph = './data/graph.pkl'

    start = '通州北苑站'
    end = '西二旗站'

    # 1. 爬取地铁的线路数据, 使用高德地图API，获取地点的坐标
    df = create_or_load_subway_location_data(beijing_subway_data_path, beijing_subway_location_data_path)
    logger.debug(df.tail())

    # 2. 获取站点之间的距离图
    graph = create_or_load_distance_graph(df, beijing_subway_site_distance_graph)
    logger.debug(graph)

    # 3. 使用 Dijkstra 计算起始站到终点站的最优路径
    costs, parents = get_stations_costs_and_parents(graph, start, end)
    print(costs, parents)
    shortest_path = dijkstra(graph, costs, parents, start, end)
    logger.info(f'从 {start} 到 {end} 的最短路径: {shortest_path}，距离为: {costs[end]}')


if __name__ == '__main__':
    run()
