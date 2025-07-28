import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sympy import rotations
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from  sklearn.metrics import  classification_report, confusion_matrix

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


##Veriye Genel Bakış #####
train.info(verbose=True)
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
train.drop(columns= ['CGPA'], inplace=True)

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


### modelleme ####
#### 2 'li objectleri labelencoder #####
label_cols = [
    'Gender', 'Working Professional or Student',
    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'
]
le = LabelEncoder()
for col in label_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

multi_cat_cols = [
     'Sleep Duration', 'Dietary Habits', 'Degree'
]
le = LabelEncoder()
for col in multi_cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


train.drop(columns = ['Name', 'City'], inplace=True)
test.drop(columns = ['Name', 'City'], inplace=True)

train['Profession'].value_counts()
train['Sleep Duration'].value_counts()
train['Dietary Habits'].value_counts()
train['Degree'].value_counts()

silinmeyecek_kategori = [
    'Less than 5 hours', 'More than 8 hours', '5-6 hours', '6-7 hours'
    '7-8 hours', '8-9 hours',
]
train['Sleep Duration'] = train['Sleep Duration'].apply(lambda x:x if x in silinmeyecek_kategori else 'Other')
test['Sleep Duration'] = test['Sleep Duration'].apply(lambda x:x if x in silinmeyecek_kategori else 'Other')

silinmeyecek_kategori_diet = [
    'Moderate', 'Unhealthy', 'Healthy'
]
train['Dietary Habits'] = train['Dietary Habits'].apply(lambda x:x if x in silinmeyecek_kategori_diet else 'Others')
test['Dietary Habits'] = test['Dietary Habits'].apply(lambda x:x if x in silinmeyecek_kategori_diet else 'Others')

degree_counts = train['Degree'].value_counts()
rare_degress = degree_counts[degree_counts < 100].index
train['Degree'] = train['Degree'].replace(rare_degress, 'Other')
test['Degree'] = test['Degree'].replace(rare_degress, 'Other')

X = train.drop(columns=['Depression'])
y = train['Depression']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('confussion matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değerler')
plt.show()

