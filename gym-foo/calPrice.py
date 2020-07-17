#购电方队列
# buyer_name = ["a1","b1","c1","d1","e1"]
# buyer_num = [[350, 1000],[430,3000],[230, 4000],[300,5000],[430, 2000]]   [电价，电量]
# #售电方队列
# seller_name = ["f1","g1","h1","i1","j1"]
# seller_num = [[150, 4000],[250, 1500],[200, 2000],[400, 3000],[300, 3000]]  [电价，电量]
# 说明：buyer_name是购电方名称列表，buyer_num是购电方申报数据列表，名称与数据按照顺序一一对应。数据里前面的是电价，后面的是电量。对于售电方同理。
import numpy as np
def rank(buyer_name, buyer_num, seller_name, seller_num):
    buyer_amount = []
    for i in buyer_num:
        buyer_amount.append(i[0])
    buyer_amount_sort = sorted(buyer_amount, reverse=True)  # 购电方按照申报电价降序排列
    buyer_number_sort = []
    buyer_name_sort = []
    buyer_num_temp = buyer_num.copy()
    buyer_name_temp = buyer_name.copy()

    for j in range(len(buyer_num)):
        for i in range(len(buyer_num_temp)):
            if buyer_num_temp[i][0] == buyer_amount_sort[0]:
                buyer_number_sort.append(buyer_num_temp[i])
                buyer_name_sort.append(buyer_name_temp[i])
                buyer_amount_sort.pop(0)
                buyer_num_temp.pop(i)
                buyer_name_temp.pop(i)
                break

    seller_amount = []
    for i in seller_num:
        seller_amount.append(i[0])
    seller_amount_sort = sorted(seller_amount)  # 售电方按照申报电价升序排列
    seller_number_sort = []
    seller_name_sort = []
    seller_num_temp = seller_num.copy()
    seller_name_temp = seller_name.copy()

    for j in range(len(seller_num)):
        for i in range(len(seller_num_temp)):
            if seller_num_temp[i][0] == seller_amount_sort[0]:
                seller_number_sort.append(seller_num_temp[i])
                seller_name_sort.append(seller_name_temp[i])
                seller_amount_sort.pop(0)
                seller_num_temp.pop(i)
                seller_name_temp.pop(i)
                break
    # return buyer_number_sort,buyer_name_sort,seller_number_sort,seller_name_sort
    match_result, clear_price = match(buyer_number_sort,buyer_name_sort,seller_number_sort,seller_name_sort)
    return match_result,clear_price

def match(buyer_number_sort, buyer_name_sort,seller_number_sort,seller_name_sort):
    buyer_number_sort_copy = buyer_number_sort.copy()
    seller_number_sort_copy = seller_number_sort.copy()
    buyer_name_sort_copy = buyer_name_sort.copy()
    seller_name_sort_copy = seller_name_sort.copy()

    # 记录匹配结果的列表,[[买方名字，卖方名字，匹配电量],[买方名字，卖方名字，匹配电量]，[……]]
    match_result = []
    # 当买方队列第一个申报电价大于卖方队列第一个申报电价，则满足匹配条件
    # 主循环 停止条件：1、不满足买方申报电价大于卖方申报电价 2、一方电量完全交易完 3、成交电量达到集中竞价交易电量规模
    while buyer_number_sort_copy[0][0] >= seller_number_sort_copy[0][0]:
        # 计算买方队列前方申报电价相同的申报电量总量
        m = 0  # 买方队列前方共有m+1个相同申报电价的成员
        buyer_amount = buyer_number_sort_copy[m][1]
        while m < (len(buyer_number_sort_copy) - 1):
            if buyer_number_sort_copy[m][0] == buyer_number_sort_copy[m + 1][0]:
                buyer_amount = buyer_amount + buyer_number_sort_copy[m + 1][1]
                m = m + 1
            else:
                break
            # 计算卖方队列前方申报电价相同的申报电量总量
        n = 0  # 卖方队列前方共有n+1个相同申报电价的成员
        seller_amount = seller_number_sort_copy[n][1]
        while n < (len(seller_number_sort_copy) - 1):
            if seller_number_sort_copy[n][0] == seller_number_sort_copy[n + 1][0]:
                seller_amount = seller_amount + seller_number_sort_copy[n + 1][1]
                n = n + 1
            else:
                break

        # 买方申报电量大于卖方申报电量，则该电量下的卖方全部完成交易
        if buyer_amount > seller_amount:
            # 进行匹配
            for j in range(n + 1):
                for i in range(m + 1):
                    match_amount = (seller_number_sort_copy[j][1] / buyer_amount) * buyer_number_sort_copy[i][1]
                    if np.isnan(match_amount):       #会为空？？？
                        match_amount = 0
                    match_amount = int(match_amount + 0.5)  # 四舍五入，取整数
                    match_result.append([buyer_name_sort_copy[i], seller_name_sort_copy[j], match_amount])

            # 更新匹配过的买方信息
            for i in range(m + 1):
                trade_amount = (seller_amount / buyer_amount) * buyer_number_sort_copy[i][1]
                if np.isnan(trade_amount):  # 会为空？？？
                    trade_amount = 0
                trade_amount = int(trade_amount + 0.5)
                buyer_number_sort_copy[i][1] = buyer_number_sort_copy[i][1] - trade_amount
            # 删除配对完成的卖方
            for j in range(n + 1):
                seller_number_sort_copy.pop(0)
                seller_name_sort_copy.pop(0)

        # 买方申报电量小于卖方申报电量，则该电量下的买方全部完成交易
        if buyer_amount < seller_amount:
            # 进行匹配
            for i in range(m + 1):
                for j in range(n + 1):
                    match_amount = (buyer_number_sort_copy[i][1] / seller_amount) * seller_number_sort_copy[j][1]
                    if np.isnan(match_amount):       #会为空？？？
                        match_amount = 0
                    match_amount = int(match_amount + 0.5)  # 四舍五入，取整数
                    match_result.append([buyer_name_sort_copy[i], seller_name_sort_copy[j], match_amount])

            # 更新匹配过的卖方信息
            for j in range(n + 1):
                trade_amount = (buyer_amount / seller_amount) * seller_number_sort_copy[j][1]
                if np.isnan(trade_amount):  # 会为空？？？
                    trade_amount = 0
                trade_amount = int(trade_amount + 0.5)
                seller_number_sort_copy[j][1] = seller_number_sort_copy[j][1] - trade_amount

            # 删除配对完成的买方
            for i in range(m + 1):
                buyer_number_sort_copy.pop(0)
                buyer_name_sort_copy.pop(0)

        # 买方申报电量等于卖方申报电量，则该电量下的买卖双方均全部完成交易
        if buyer_amount == seller_amount:
            # 进行匹配
            for j in range(n + 1):
                for i in range(m + 1):
                    match_amount = (seller_number_sort_copy[j][1] / buyer_amount) * buyer_number_sort_copy[i][1]
                    # print(match_amount.type())
                    if np.isnan(match_amount):       #会为空？？？
                        match_amount = 0
                    match_amount = int(match_amount + 0.5)  # 四舍五入，取整数
                    match_result.append([buyer_name_sort_copy[i], seller_name_sort_copy[j], match_amount])
            # 删除配对完成的买方
            for i in range(m + 1):
                buyer_number_sort_copy.pop(0)
                buyer_name_sort_copy.pop(0)
            # 删除配对完成的卖方
            for j in range(n + 1):
                seller_number_sort_copy.pop(0)
                seller_name_sort_copy.pop(0)

        # 判断买方交易队列和卖方交易队列，若一方为空，则一方交易完，跳出循环。
        if((len(buyer_number_sort_copy) == 0) or (len(seller_number_sort_copy) == 0)) :
            break

    if(len(match_result)!=0):
        buyer_index = buyer_name_sort.index(match_result[-1][0])
        seller_index = seller_name_sort.index(match_result[-1][1])
        buyer_last_price = buyer_number_sort[buyer_index][0]
        seller_last_price = seller_number_sort[seller_index][0]
        clear_price = round((buyer_last_price + seller_last_price) / 2, 2)  # 出清电价保留两位小数
    else:clear_price = 0
    return match_result,clear_price

def select_clear_amount(name,match_result):
    amount = 0
    for i in match_result:
        if name in i :
            amount+=i[2]
    return amount

def calculate_total_amount(match_result):
    amount = 0
    for i in match_result:
        amount+=i[2]
    return amount
# 程序使用示例：

# 传入购电方
# buyer_name = ['陕西秦电配售电有限责任公司', '大唐陕西能源营销有限公司','陕西榆林能源集团售电有限公司','陕西深电能售电有限公司','陕西洁能售电有限公司','郑州沃特节能科技股份有限公司','陕西盈智能源科技有限公司']
# buyer_num = [[354.9, 74559],[354.5, 58484],[353.8,31662],[353.3,25406],[352.8,20688],[351.9,20066],[350.5,17432]]
# # 传入售电方
# seller_name = ['陕西华电蒲城发电有限责任公司', '陕西渭河发电有限公司','陕西宝鸡第二发电有限责任公司','陕西华电杨凌热电有限公司','陕西清水川发电有限公司']
# seller_num = [[345.3, 63120], [346.2, 57021],[347.4,50117],[348.8,32016],[349.7,26163]]
# # 进行匹配
# match_result,clear_price = rank(buyer_name,buyer_num,seller_name,seller_num)
# # 计算每一个售电方的总成交电量
#
# for i in seller_name:
#     amount = select_clear_amount(i,match_result)
#     value = str(amount)
#     print(i+"  匹配电量  "+value)
# # 计算每一个购电方的总成交电量
# for i in buyer_name:
#     amount = select_clear_amount(i,match_result)
#     value = str(amount)
#     print(i+"  匹配电量  "+value)
#
