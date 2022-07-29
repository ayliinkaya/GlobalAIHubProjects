import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
df = pd.read_csv("NetflixOriginals.csv", encoding = "ISO-8859-1")
df.head()
df.shape

df["Language"].value_counts()

# ● Veri setine göre uzun soluklu filmler hangi dilde oluşturulmuştur? Görselleştirme yapınız.

movie_runtime = df.groupby("Language").agg({"Runtime": "mean"}).sort_values(["Runtime"], ascending=False)
movie_runtime.reset_index(inplace=True)
movie_runtime

plt.figure(figsize=(12,12))
plt.barh(movie_runtime.Language,movie_runtime.Runtime,label="Runtime")
plt.ylabel("Language")
plt.xlabel("Runtime")
plt.legend()
plt.title("Grouping of Movie Runtimes by Language")
plt.show()

# ● 2019 Ocak ile 2020 Haziran tarihleri arasında 'Documentary' türünde çekilmiş
# filmlerin IMDB değerlerini bulup görselleştiriniz.

for col in df.columns:
    if "Premiere" in col:
        df[col] = df[col].apply(pd.to_datetime)

doc_movie = df.loc[(df["Premiere"] > "2019-01-31") & (df["Premiere"] < "2020-06-01") & (df["Genre"] == "Documentary")].sort_values(["Runtime"], ascending=False)
doc_movie.reset_index(inplace = True)

df.rename(columns = {'IMDB Score':'IMDB_Score'}, inplace = True)

fig=plt.figure(figsize=(12,12))
plt.barh(doc_movie.Title, doc_movie.IMDB_Score, label="IMDB_Score")
plt.ylabel("Title")
plt.xlabel("IMDB_Score")
plt.legend()
plt.title("IMDB scores of 'Documentary' movies between January 2019 and June 2020")
plt.show()

# ● İngilizce çekilen filmler içerisinde hangi tür en yüksek IMDB puanına sahiptir?

eng_movies = df.loc[df["Language"] == "English"]
eng_movies[["Title", "Genre", "Language", "IMDB_Score"]].sort_values(["IMDB_Score"], ascending=False)[0:1]

# ● 'Hindi' Dilinde çekilmiş olan filmlerin ortalama 'runtime' suresi nedir?

df.loc[df["Language"] == "Hindi"].agg({"Runtime": "mean"})

# ● 'Genre' Sütunu kaç kategoriye sahiptir ve bu kategoriler nelerdir?
# Görselleştirerek ifade ediniz.

df["Genre"].value_counts()
df["Genre"].unique()
df["Genre"].nunique()

df["Genre"].value_counts().plot(kind='bar')
plt.show()

# ● Veri setinde bulunan filmlerde en çok kullanılan 3 dili bulunuz.

df["Language"].value_counts().head(3)

# ● IMDB puanı en yüksek olan ilk 10 film hangileridir?

df.groupby("Title").agg({"IMDB_Score": "max"}).sort_values(["IMDB_Score"], ascending=False).head(10).reset_index()

# ● IMDB puanı ile 'Runtime' arasında nasıl bir korelasyon vardır? İnceleyip
# görselleştiriniz.

corr = df["Runtime"].corr(df["IMDB_Score"])
corr = np.round(df.corr(),2)
corr

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(ax=ax, data=corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
ax.set_title("Correlation Matrix Heatmap")
plt.show()

# ● IMDB Puanı en yüksek olan ilk 10 'Genre' hangileridir? Görselleştiriniz.

genre_imdb_top10 = df.groupby("Genre").agg({"IMDB_Score": "max"}).sort_values(["IMDB_Score"], ascending=False).head(10).reset_index()

plt.figure(figsize=(30,6))
plt.bar(genre_imdb_top10.Genre,
        genre_imdb_top10.IMDB_Score,
        width=0.5,
        color='#e60000',
        label="IMDB_Score")
plt.ylabel("IMDB_Score")
plt.xlabel("Genre")
plt.legend()
plt.title("Top 10 IMDB Scores with Genre")
plt.show()

# ● 'Runtime' değeri en yüksek olan ilk 10 film hangileridir? Görselleştiriniz.

title_runtimetop10 = df.groupby("Title").agg({"Runtime": "max"}).sort_values(["Runtime"], ascending=False).head(10).reset_index()

plt.figure(figsize=(30,6))
plt.bar(title_runtimetop10.Title,
        title_runtimetop10.Runtime,
        width=0.5,
        color='#e60000',
        label="Runtime")
plt.ylabel("Runtime")
plt.xlabel("Title")
plt.legend()
plt.title("Top 10 Runtime with Title")
plt.show()

# ● Hangi yılda en fazla film yayımlanmıştır? Görselleştiriniz. ???????????????????

df["Year"] = df["Premiere"].dt.year
year = df["Year"].value_counts(sort=True)
year = year.to_frame().reset_index()
year.columns = ["Year", "Count"]
print(year)

plt.figure(figsize=(10,8))
plt.bar(year.Year,year.Count,
        width=0.5,
        color='#0080ff',
        label="Count")
plt.ylabel("Count")
plt.xlabel("Year")
plt.legend()
plt.title('Most Watched Movies vs Years', fontsize=15)
plt.show()

# ● Hangi dilde yayımlanan filmler en düşük ortalama IMBD puanına sahiptir?
# Görselleştiriniz.

imdb_lowest = df.groupby("Language").agg({"IMDB_Score": "mean"}).sort_values(["IMDB_Score"], ascending=True)[0:1]
lowest_imdb_score = imdb_lowest(df)
lowest_imdb_score.reset_index(inplace=True)

plt.figure(figsize=(30,6))
plt.bar(lowest_imdb_score.Language,
        lowest_imdb_score.IMDB_Score,
        width=0.5,
        color='#e60000',
        label="IMDB_Score")
plt.ylabel("IMDB_Score")
plt.xlabel("Language")
plt.legend()
plt.title("Lowest IMDB rated movies by language")
plt.show()

def lowest_imdb_score(dataframe):
    dataframe = dataframe.groupby("Language").agg({"IMDB_Score": "mean"}).sort_values(by="IMDB_Score", ascending=True)
    runtime = dataframe.head(10)
    return(runtime)

lowest_imdb = lowest_imdb_score(df)
lowest_imdb.reset_index(inplace=True)
lowest_imdb

plt.figure(figsize=(30,6))
plt.bar(lowest_imdb.Language,
        lowest_imdb.IMDB_Score,
        width=0.5,
        color='#e60000',
        label="IMDB_Score")
plt.ylabel("IMDB_Score")
plt.xlabel("Language")
plt.legend()
plt.title("Lowest IMDB rated movies by language")
plt.show()

# ● Hangi yılın toplam "runtime" süresi en fazladır?

df.groupby("Year").agg({"Runtime": "sum"}).sort_values(["Runtime"], ascending=False)[0:1]

# ● Her bir dilin en fazla kullanıldığı "Genre" nedir?

df.groupby("Language").agg({"Genre": "max"}).reset_index()


# ● Veri setinde outlier veri var mıdır? Açıklayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):

    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Runtime")
outlier_thresholds(df, "IMDB_Score")

# Aykırı değer olup olmadığını kontrol edelim:

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Runtime")
check_outlier(df, "IMDB_Score")
check_outlier(df, "Premiere")

for col in num_cols:
    print(col, check_outlier(df, col))

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "IMDB_Score", True)
grab_outliers(df, "Runtime", True)


sns.boxplot(x=df["IMDB_Score"])
plt.show()

sns.boxplot(x=df["Runtime"])
plt.show()

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit # alt sınırdan küçük değerler alt sınıra eşitlendi.
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit # üst sınırdan büyük aykırı değerler üst sınıra eşitlendi

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Aykırı değer problemi baskılama yöntemi ile çözülmüş oldu. Bu durum grafikler üzerinde de incelenebilir:

sns.boxplot(x=df["IMDB_Score"])
plt.show()

sns.boxplot(x=df["Runtime"])
plt.show()