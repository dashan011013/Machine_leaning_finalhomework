import numpy as np
import pandas as pd
import time
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from pandas import Timestamp

# 因为UserBehavior.csv文件很大，有3.41G，所以此步骤执行完一次后不再执行，为了交作业的时候不把没必要的大文件交上去。
# df = pd.read_csv('UserBehavior/UserBehavior.csv', header=None,
#                  names=['user_id', 'item_id', 'category_id', 'behavior_type',
#                         'time_stamp'])  # header=None 让第一行不做列索引，让name后的做
# print("ok")
# 随机抽取100W条数据
# data = df.take(indices=np.random.permutation(df.shape[0]), axis=0)[0:1000000]
# 保存到本地
# data.to_csv('./UserBehavior/UserBehavior_new.csv')
# print("ok")
# 读取本地的100W数据
df = pd.read_csv('UserBehavior/UserBehavior_new.csv')

# print("ok")
# df.drop(labels='Unnamed: 0', axis=1, inplace=True)  # 删除没用的'Unnamed: 0'列
# df.head()

# 查看是否存在重复的行数据
# print((df.duplicated()).sum())  # duplicated 是看有没有重复值 返回的是一组布尔值，再将之累加 看有多少个true（true 为1  false 为0）

# 查看列中是否存在缺失数据
# print(df.isnull().any(axis=0))

# 将时间戳转换为日期格式
df['time_stamp'] = pd.to_datetime(df['time_stamp'], unit='s', origin=pd.Timestamp('1970-01-01'))

# 转换'time_stamp'列的数值类型为datatime
df['time_stamp'] = pd.to_datetime(df['time_stamp'])

# 添加一列为月份
df['month'] = df['time_stamp'].dt.month

# # 添加一列为年份
# df['year'] = df['time_stamp'].dt.year

# # 查看数据的时间范围，如有异常值将其删除
# # 发现极大部分数据都是17年，其他都是2025年、1927年等离谱数据 这里只保留17年的数据
# print(df['year'].value_counts())

# 筛选出17年的数据
ex = (df['time_stamp'] >= '2017-01-01') & (df['time_stamp'] <= '2017-12-31')
# 将17年的数据取出 赋值给原数据
df = df.loc[ex]
# 看一下 最大最小日期是否符合要求
# df['time_stamp'].max(), df['time_stamp'].min()
#
# print(df['time_stamp'].max())
# print(df['time_stamp'].min())


# # 对所有用户的不同购买行为进行数量统计且求得不同购买行为的百分比,以柱状图进行展示
# # 针对同一用户多次PV 这里使用nunique 进行去重
# s_persent = df.groupby(by='behavior_type')['user_id'].nunique() / df.groupby(by='behavior_type')[
#     'user_id'].nunique().sum() * 100
# plt.bar(s_persent.index, s_persent.values)
# plt.show()


# 发现用户点击量占据83%，而购买量仅占所有数据的3.3%,，用户从浏览到购买的转化#率只有2%，那是什么原因导致的转化率低呢？
# 分析出每个用户对商品的不同行为
df.head()
one_hot_df = pd.get_dummies(
    # one_hot编码 针对某些不是数值型的数据 进行数据汇总使用get_dummies  将之转换成数据型数据 0 和1  然后以行索引 确定0/1 所代表的的信息
    df['behavior_type'])

# 将用户ID和商品ID和one_hot_df 横向拼接
user_item_behavior_df = pd.concat((df[['user_id', 'item_id']], one_hot_df), axis=1)

user_item_behavior_df.head()

# 分析出每个用户对商品的不同行为次数的汇总
pv_sum = user_item_behavior_df.groupby(by='user_id')['pv'].sum()
buy_sum = user_item_behavior_df.groupby(by='user_id')['buy'].sum()
cart_sum = user_item_behavior_df.groupby(by='user_id')['cart'].sum()
fav_sum = user_item_behavior_df.groupby(by='user_id')['fav'].sum()

user_behavior_total_df = DataFrame(data=[pv_sum, buy_sum, cart_sum, fav_sum]).T  # T转置让行变列 列变行
user_behavior_total_df.head()

# 1.点击量:所有用户的总点击量
pv_count = user_behavior_total_df['pv'].sum()
print(pv_count)  # 895841

# 2.点击--购买：用户点击后无加购和收藏的情况下直接参与购买的行为统计
pv_buy_count = user_behavior_total_df.query('pv > 0 & cart ==0 & fav == 0 & buy > 0').shape[0]
print(pv_buy_count)  # 10062

# 3.点击--加购：点击后，无收藏情况下的加购行为
pv_cart_count = user_behavior_total_df.query('pv > 0 & cart > 0 & fav == 0').shape[0]
print(pv_cart_count)  # 31429

# 4.点击--加购--购买：点击后无收藏情况下的加购和购买行为
pv_cart_buy_count = user_behavior_total_df.query('pv > 0 & cart > 0 & buy > 0 & fav ==0').shape[0]
print(pv_cart_buy_count)  # 962

# 5.点击--收藏：点击后，无加购情况下的收藏行为
pv_fav_count = user_behavior_total_df.query('pv > 0 & fav > 0 & cart == 0').shape[0]
print(pv_fav_count)  # 16534

# 6.点击--收藏--购买：点击后，无加购情况下的收藏和购买行为
pv_fav_buy_count = user_behavior_total_df.query('pv > 0 & fav > 0 & buy > 0 & cart ==0').shape[0]
print(pv_fav_buy_count)  # 438

# 7.点击--收藏+加购：点击后的收藏和加购行为
pv_fav_cart_count = user_behavior_total_df.query('pv > 0 & fav > 0 & cart > 0').shape[0]
print(pv_fav_cart_count)  # 1020

# 8.点击--收藏+加购 -- 购买：点击后的收藏加购和购买的行为
pv_fav_cart_buy_count = user_behavior_total_df.query('pv > 0 & fav > 0 & cart > 0 & buy > 0').shape[0]
print(pv_fav_cart_buy_count)  # 33

# 9.点击--流失：点击后无购买无加购无收藏的行为
pv_loss_count = user_behavior_total_df.query('pv > 0 & buy ==0 & cart == 0 & fav ==0 ').shape[0]
print(pv_loss_count)  # 434216
#
# print("--------------------------------------")
# 1.直接购买转化率：点击–购买 / 点击量
print(pv_buy_count / pv_count)  # 0.01123190387579939

# 2.加购购买转换率：点击 => 加购 + 购买 / 点击 => 加购
print(pv_cart_buy_count / pv_cart_count)  # 0.01123190387579939

# 3.收藏购买转换率：点击 => 收藏 => 购买 / 点击 => 收藏
print(pv_fav_buy_count / pv_fav_count)  # 0.02649086730373775

# 4.加购收藏购买转换率：点击 => 加购 + 收藏 => 购买 / 点击 => 加购 + 收藏
print(pv_fav_cart_buy_count / pv_fav_cart_count)  # 0.03235294117647059

# 5.流失率：点击–流失 / 点击量
print(pv_loss_count / pv_count)  # 0.4847020844100683


# 直接购买转化率低于加购和收藏等行为之后的综合转换率，因此需要从产品交互界面、#营销机制等方面让用户去多加购，多收藏。
# 转化率低的原因分析：
# 提出假设：推荐机制不合理，给用户推荐的都是不喜欢的商品，造成转化率低
# 这里可以通过分析高浏览量商品与高购买量商品之间是否存在高度重合，如果是的，那就说明推荐的商品是用户喜欢，假设就不成立，如果不是则证明假设成立。

# 分析出点击量前10的商品
pv_sum_item_10_s = user_item_behavior_df.groupby(by='item_id')['pv'].sum().sort_values().tail(10)
# print(pv_sum_item_10_s)

# 购买量前10的商品
buy_sum_item_10_s = user_item_behavior_df.groupby(by='item_id')['buy'].sum().sort_values().tail(10)
# print(buy_sum_item_10_s)

# 查看点击量高且购买量也高的商品类别个数
# print(buy_sum_item_10_s.append(pv_sum_item_10_s).index.value_counts())  # 把两个series 拼一起 看那个商品出现在两个series里

i = 0
# 计算点击量前10的商品的购买量

pv_10_buy = []
for index in pv_sum_item_10_s.index:
    buy_count = user_item_behavior_df.loc[user_item_behavior_df['item_id'] == index]['buy'].sum()
    dic = {
        'item_id': index,
        'buy_count': buy_count
    }
    pv_10_buy.append(dic)
    print(pv_10_buy[i])
    i = i + 1

i = 0
# 计算购买量前10的点击量
pv_sum_list = []
for index in buy_sum_item_10_s.index:
    pv_sum = user_item_behavior_df.loc[user_item_behavior_df['item_id'] == index]['pv'].sum()
    dic = {
        'item_id': index,
        'pv_sum': pv_sum
    }
    pv_sum_list.append(dic)
    print(pv_sum_list[i])
    i = i + 1
