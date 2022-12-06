import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# def _lay_theo_idx(df, column):
#     unique_colunm = list(pd.unique(df[column]))
#     for idx, name in enumerate(unique_colunm):
#         # get index
#         index = df.index[df[column] == name].tolist()
#         df.loc[index,column] = int(idx)
#     return df
#####Sử dụng pandas để đọc dữ liệu
df = pd.read_csv('Case_study_CarPrice_Assignment.csv')

#print(df.head())
# print(df.describe())
# print(df.isna().sum())
df['BrandName'] = df.apply(lambda x:str(x['CarName']).split(" ")[0],axis=1).reset_index(drop=True)
##### Tìm mối liên hệ giữa hãng xe và tên xe, phát hiện và sửa sai dữ liệu
#print(pd.unique(df['CarName']))
## >>>Tên hãng xe nằm trong tên xe 
# print(df.describe())

#####các thuộc tính gây ảnh hưởng tới giá
data = df[['symboling','fueltype','doornumber','carbody','carlength','carwidth','carheight','enginetype','fuelsystem','horsepower','peakrpm','citympg','highwaympg','BrandName','price']]
#print(data.info())


#### Xem xét kiểu dữ liệu của các thuộc tính, thực hiện chuyển đổi về đúng kiểu
# print(df.info())
# print(df.isna().sum())
#>>> data không có dữ liệu khuyết thiếu
data['fueltype'] = data['fueltype'].astype('category').cat.codes
data['doornumber'] = data['doornumber'].astype('category').cat.codes
data['carbody'] = data['carbody'].astype('category').cat.codes
data['enginetype'] = data['enginetype'].astype('category').cat.codes
data['fuelsystem'] = data['fuelsystem'].astype('category').cat.codes
data['BrandName'] = data['BrandName'].astype('category').cat.codes

# print(data.info())
# print(data.describe())




#### heat map
print(data.corr())
relation =data.corr()
relation_index=relation.index
sns.heatmap(data,yticklabels=False)
sns.heatmap(data[relation_index].corr(),annot=True)
plt.show()


x = data[['symboling','fueltype','doornumber','carbody','carlength','carwidth','carheight','enginetype','fuelsystem','horsepower','peakrpm','citympg','highwaympg','BrandName','price']]
y = data[['price']]

x_train, x_test, y_train, y_test = train_test_split( x,y, test_size=0.2, random_state=42)

### huấn luyện với 2 mô hình LinearRegression và mô hình RandomForestRegressor
#LinearRegression
linear = LinearRegression()
linear.fit(x_train,y_train)
y_prediction = linear.predict(x_test) 
score = r2_score(y_test,y_prediction)

print('MAPE',mean_absolute_percentage_error(y_test,y_prediction))
print('R2-score is ',score) 


#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
Rf_rg = RandomForestRegressor()
Rf_rg.fit(x_train,y_train)
y_prediction = Rf_rg.predict(x_test) 
score = r2_score(y_test,y_prediction)
print('MAPE',mean_absolute_percentage_error(y_test,y_prediction))
print('R2-score is ',score) 


#>>>> Mô hình LinearRegression có R2-score và phương sai thấp hơn 