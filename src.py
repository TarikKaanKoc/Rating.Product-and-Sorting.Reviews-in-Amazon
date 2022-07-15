#############################################
#                                           #
#  RatingProduct & SortingReviewsin Amazon  #
#                                           #
#############################################

#------------------------------------------------------------------------------------------
# Bussines Problem:

''''
One of the most important problems in e-commerce is the correct calculation of the points
given to the products after sales. The solution to this problem means providing greater
customer satisfaction for the e-commerce site, prominence of the product for the sellers
and a seamless shopping experience for the buyers. Another problem is the correct ordering
of the comments given to the products. The prominence of misleading comments will cause both
financial loss and loss of customers. In the solution of these 2 basic problems, while the
e-commerce site and the sellers will increase their sales, the customers will complete the
purchasing journey without any problems.
'''


# Dataset Story:

'''
This dataset containing Amazon Product Data includes product categories and various metadata.
The product with the most comments in the electronics category has user ratings and comments.


# CSV File Size: 71.9MB
# Features: 12
# Row: 4915   
'''

# Features:

'''
- reviewerID --> User Id
- asin --> Product Id
- reviewerName --> User name 
- helpful --> Useful Evaluation Degree
- reviewText --> Evaluation
- overall --> Product Rating
- summary --> Evaluation Summary
- unixReviewTime --> Evaluation Time
- reviewTime --> Evaluation Time {RAW}
- day_diff --> Number of days since assessment
- helpful_yes --> The number of times the evaluation was found useful
- total_vote --> Number of votes given to the evaluation

'''

#------------------------------------------------------------------------------------------

import pandas as pd
import math
import scipy.stats as st


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import warnings
warnings.filterwarnings("ignore")


df_ = pd.read_csv("amazon_review.csv")

df = df_.copy()

df.head()

df.shape # OUT: (4915, 12)

df.info()
'''
 0   reviewerID      4915 non-null   object 
 1   asin            4915 non-null   object 
 2   reviewerName    4914 non-null   object 
 3   helpful         4915 non-null   object 
 4   reviewText      4914 non-null   object 
 5   overall         4915 non-null   float64
 6   summary         4915 non-null   object 
 7   unixReviewTime  4915 non-null   int64  
 8   reviewTime      4915 non-null   object 
 9   day_diff        4915 non-null   int64  
 10  helpful_yes     4915 non-null   int64  
 11  total_vote      4915 non-null   int64  
'''

df.describe().T
'''
                    count             mean            std              min              25%              50%              75%              max
overall        4915.00000          4.58759        0.99685          1.00000          5.00000          5.00000          5.00000          5.00000
unixReviewTime 4915.00000 1379465001.66836 15818574.32275 1339200000.00000 1365897600.00000 1381276800.00000 1392163200.00000 1406073600.00000
day_diff       4915.00000        437.36704      209.43987          1.00000        281.00000        431.00000        601.00000       1064.00000
helpful_yes    4915.00000          1.31109       41.61916          0.00000          0.00000          0.00000          0.00000       1952.00000
total_vote     4915.00000          1.52146       44.12309          0.00000          0.00000          0.00000          0.00000       2020.00000
'''


# If there is more than one product, we will observe the average score for each product with Groupby.
df['asin'].value_counts()

# Since it is a single product, I directly take the average and observe it.
df['overall'].mean()
# OUT: 4.5875


# Calculate the weighted average score by date
df["reviewTime"] = pd.to_datetime(df["reviewTime"], dayfirst=True)
current_date = pd.to_datetime("2021-02-12")

df["days"] = (current_date - df["reviewTime"]).dt.days

df.days.describe().T
'''
- count   4915.00
- mean    2695.37
- std      209.44
- min     2259.00
- 25%     2539.00
- 50%     2689.00
- 75%     2859.00
- max     3322.00
'''

df.head()
''''
      reviewerID        asin  reviewerName helpful                                         reviewText  overall                                 summary  unixReviewTime reviewTime  day_diff  helpful_yes  total_vote  days
0  A3SBTW3WS4IQSN  B007WTAJTO           NaN  [0, 0]                                         No issues.     4.00                              Four Stars      1406073600 2014-07-23       138            0           0  2396
1  A18K1ODH1I2MVB  B007WTAJTO          0mie  [0, 0]  Purchased this for my device, it worked as adv...     5.00                           MOAR SPACE!!!      1382659200 2013-10-25       409            0           0  2667
2  A2FII3I2MBMUIA  B007WTAJTO           1K3  [0, 0]  it works as expected. I should have sprung for...     4.00               nothing to really say....      1356220800 2012-12-23       715            0           0  2973
3   A3H99DFEG68SR  B007WTAJTO           1m2  [0, 0]  This think has worked out great.Had a diff. br...     5.00  Great buy at this price!!!  *** UPDATE      1384992000 2013-11-21       382            0           0  2640
4  A375ZM4U047O79  B007WTAJTO  2&amp;1/2Men  [0, 0]  Bought it with Retail Packaging, arrived legit...     5.00                        best deal around      1373673600 2013-07-13       513            0           0  2771
'''

q_1 = df["days"].quantile(0.25)
q_1 # OUT: 2539
q_2 = df["days"].quantile(0.50)
q_2 # OUT: 2689
q_3 = df["days"].quantile(0.75)
q_3 # OUT: 2859


# Calculating the weighted score based on the Q1 - Q2 - Q3 values:

print(df.loc[(df["days"] <= 2539), "overall"].mean())
print(df.loc[(df["days"] > 2539) & (df["day_diff"] <= 2689), "overall"].mean())
print(df.loc[(df["days"] > 2689) & (df["day_diff"] <= 2859), "overall"].mean())
print(df.loc[(df["days"] > 2859), "overall"].mean())
''''
OUT 1 --> 4.6957928802588995
OUT 2 --> 4.551236749116608
OUT 3 --> 4.508957654723127
OUT 4 --> 4.4462540716612375
'''

# Specifying 20 reviews for the product that will be displayed on the product detail page
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df.head()

''''
The aim of the project is to analyze ratings and sort reviews, 
therefore, half of the variables can be eliminated.
'''

# Required Features:
df = df[["overall", "reviewTime", "day_diff", "helpful_yes", "helpful_no","total_vote"]]
df.head()

'''
   overall reviewTime  day_diff  helpful_yes  total_vote
0  4.00000 2014-07-23       138            0           0
1  5.00000 2013-10-25       409            0           0
2  4.00000 2012-12-23       715            0           0
3  5.00000 2013-11-21       382            0           0
4  5.00000 2013-07-13       513            0           0
'''

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()

# RESULT :
df.sort_values("wilson_lower_bound", ascending=False).head(20)

''''
     overall reviewTime  day_diff  helpful_yes  helpful_no  total_vote  score_pos_neg_diff  score_average_rating  wilson_lower_bound
2031  5.00000 2013-01-05       702         1952          68        2020                1884               0.96634             0.95754
3449  5.00000 2012-09-26       803         1428          77        1505                1351               0.94884             0.93652
4212  1.00000 2013-05-08       579         1568         126        1694                1442               0.92562             0.91214
317   1.00000 2012-02-09      1033          422          73         495                 349               0.85253             0.81858
4672  5.00000 2014-07-03       158           45           4          49                  41               0.91837             0.80811
1835  5.00000 2014-02-28       283           60           8          68                  52               0.88235             0.78465
3981  5.00000 2012-10-22       777          112          27         139                  85               0.80576             0.73214
3807  3.00000 2013-02-27       649           22           3          25                  19               0.88000             0.70044
4306  5.00000 2012-09-06       823           51          14          65                  37               0.78462             0.67033
4596  1.00000 2012-09-22       807           82          27         109                  55               0.75229             0.66359
315   5.00000 2012-08-13       847           38          10          48                  28               0.79167             0.65741
1465  4.00000 2014-04-14       238            7           0           7                   7               1.00000             0.64567
1609  5.00000 2014-03-26       257            7           0           7                   7               1.00000             0.64567
4302  5.00000 2014-03-21       262           14           2          16                  12               0.87500             0.63977
4072  5.00000 2012-11-09       759            6           0           6                   6               1.00000             0.60967
1072  5.00000 2012-05-10       942            5           0           5                   5               1.00000             0.56552
2583  5.00000 2013-08-06       489            5           0           5                   5               1.00000             0.56552
121   5.00000 2012-05-09       943            5           0           5                   5               1.00000             0.56552
1142  5.00000 2014-02-04       307            5           0           5                   5               1.00000             0.56552
1753  5.00000 2012-10-22       777            5           0           5                   5               1.00000             0.56552
'''

