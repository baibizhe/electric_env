import  numpy as np
seller_volum = np.random.uniform(30000, 80000, [1, 3])[0]  # 随机设置了卖方容量已用测试
max_seller_voloum = 90000
min_seller_voloum = 20000
buyer_voloum = np.random.uniform(30000, 80000, [1, 3])[0]
state = np_random.uniform(min_seller_price, max_seller_price,num_of_seller)
max_buyer_price=300
max_seller_price=400
min_seller_price=300
min_buyer_price=450
num_of_seller =3
num_of_buyer = 3
def _get_buyer_random_price():
    return np.random.uniform(min_buyer_price, max_seller_price, 3)

buyer_price = _get_buyer_random_price()
random_name = ["a","b","c"]
buyer_num = [[buyer_price[i],buyer_voloum[i]] for i in range(num_of_buyer)] #构建买方的价格和容量
seller_num = [[[i],self.seller_volum[i]] for i in range(self.num_of_seller)] #构建卖方的价格和容量
