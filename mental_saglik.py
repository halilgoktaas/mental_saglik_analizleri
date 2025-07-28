import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sympy import rotations

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


##Veriye Genel Bakış #####
train.info()
test.info()
train.describe()
train.shape

test.isnull().sum()
train.isnull().sum()
train.isnull().mean() * 100
bos_veriler = train.isnull().sum()
bos_veriler.sort_values(ascending = True)
### Veri Ön İşleme####
kategorik_veriler = train.select_dtypes(include = 'object').columns
train[kategorik_veriler].isnull().sum()
kategorik_veriler_test = test.select_dtypes(include='object').columns
test[kategorik_veriler_test].isnull().sum()
### Profession Ön İşleme####
train['Profession'].value_counts()
train['Profession'].isnull().mean() * 100
profession_count = train['Profession'].value_counts()
rare_profession_count = profession_count[profession_count < 100].index
train['Profession'] = train['Profession'].replace(rare_profession_count, 'Other')
train['Profession'].fillna('Unknown', inplace=True)
test['Profession'] = test['Profession'].replace(rare_profession_count, 'Other')
test['Profession'].fillna('Unknown', inplace=True)

###Dietary Habits Ön İşleme ###
train['Dietary Habits'].value_counts()
train['Dietary Habits'].isnull().mean() * 100
train['Dietary Habits'].fillna(train['Dietary Habits'].mode()[0], inplace= True)
test['Dietary Habits'].fillna(test['Dietary Habits'].mode()[0], inplace= True)

##Degree Ön İşleme ###
train['Degree'].fillna(train['Degree'].mode()[0], inplace = True)
test['Degree'].fillna(test['Degree'].mode()[0], inplace = True)

#Sayısal Değerlere Bakış ####
sayisal_veriler = train.select_dtypes(include='float64').columns
sayisal_veriler_test = test.select_dtypes(include='float64').columns
train[sayisal_veriler].isnull().sum()
test[sayisal_veriler_test].isnull().sum()

##academic_pressure###
train['Academic Pressure'].isnull().mean() * 100
train.drop(columns = ['Academic Pressure'], inplace = True)
test.drop(columns = ['Academic Pressure'], inplace = True)

##Work Pressure###
train['Work Pressure'].isnull().mean() * 100
train['Work Pressure'].value_counts()
train['Work Pressure']=train['Work Pressure'].fillna(train['Work Pressure'].median())
test['Work Pressure']=test['Work Pressure'].fillna(test['Work Pressure'].median())

##CGPA###
train['CGPA'].isnull().mean() * 100
test.drop(columns= ['CGPA'], inplace=True)

##Study Satisfaction ###
train['Study Satisfaction'].isnull().mean() * 100
train.drop(columns = ['Study Satisfaction'], inplace= True)
test.drop(columns = ['Study Satisfaction'], inplace= True)

##Job Satisfaction ###
train['Job Satisfaction'].isnull().mean() * 100
train['Job Satisfaction'].value_counts()
train['Job Satisfaction'] = train['Job Satisfaction'].fillna(train['Job Satisfaction'].median())
test['Job Satisfaction'] = test['Job Satisfaction'].fillna(test['Job Satisfaction'].median())

##Financial Stress ##
train['Financial Stress'].isnull().mean() * 100
train['Financial Stress'].value_counts()
train['Financial Stress'] = train['Financial Stress'].fillna(train['Financial Stress'].median())
test['Financial Stress'] = test['Financial Stress'].fillna(test['Financial Stress'].median())

##EDA ve Görselleştirme###
sns.countplot(data= train, x = 'Depression')
plt.title('Hedef değişken')
plt.xlabel('Depresyon')
plt.ylabel('Sayı')
plt.tight_layout()
plt.show()

## kategorik değişkenlerin grafikleri ###
plt.figure(figsize=(16,8))
sns.countplot(data = train, x = 'Sleep Duration', hue='Depression')
plt.title('Uykunun Depresyon Üzerine Etkisi')
plt.xlabel('Uyku Süresi')
plt.ylabel('Depresyon')
plt.xticks(rotation = 90)
plt.show()

kategorik_veriler = train.select_dtypes(include = 'object').columns.drop('Name')
for col in kategorik_veriler:
    plt.figure(figsize=(12,6))
    sns.countplot(data = train, x=col,hue='Depression')
    plt.title(f'{col} depresyona etkisi')
    plt.xticks(rotation= 90)
    plt.tight_layout()
    plt.show()

### sayısal değişkenlerle depresyon analizi ####
sayisal_veriler = train.select_dtypes(include = ['float64', 'int64']).columns.drop('Depression')
for col in sayisal_veriler:
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Depression', y=col, data =train)
    plt.title(f'{col} depresyona etkisi')
    plt.xticks(rotation= 45)
    plt.tight_layout()
    plt.show()