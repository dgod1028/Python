import numpy as np
import pandas as pd
import os
import time
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from multiprocessing import Pool
import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

cache_english_stopwords=set(stopwords.words('english'))

os.chdir("d:/study/sentiment")

POS_LIST = "positive.txt"
NEG_LIST = 'negative.txt'
TEXT = "realDonaldTrump_conversation.xlsx"
TEXT_COL = 4   ## EXCEL的第几列是文本
TIME_COL = 11  ## 时间列
LIKE_COL = 9   ## Like列
INTERVAL= "D"  ## 可选择 M, H, D
AFINN = "AFINN-111.txt"

class Sentiment:
    def __init__(self,pos_dict,neg_dict,debug = True , prob = False):
        """
        :param pos_dict : positive words dictionary(txt)
        :param neg_dict : negative words dictionary(txt)
        :param prob     : If dictionary have probability. If so, input pos_dict and neg_dict should be a list file.
                                                           If prob = True, need to input a matrix (n * 2)
        """

        print('正在载入词典:   ',end='\t')
        if prob:  ## 有概率的词典，用的是 AFINN-111
            self.sen_dict = defaultdict()
            self.sen_df = pd.read_csv('AFINN-111.txt', header=None, sep="\t")
            for i in range(self.sen_df.shape[0]):
                # data类型是
                # abandon   2
                # balabala  1   <- 第一列为key, 第二列为value
                self.sen_dict[self.sen_df.iloc[i,0]] = self.sen_df.iloc[i,1]

        else:    ## 无概率的词典，用的是单位的
            assert isinstance(pos_dict, str), 'pos_dict must be a path of text data (txt format)'
            assert isinstance(neg_dict, str), 'neg_dict must be a path of text data (txt format)'
            self.pos_dict = set(self.read_txt(POS_LIST))
            self.neg_dict = set(self.read_txt(NEG_LIST))
        print('载入完毕')
        self.debug = debug
        self.prob = prob

    def read_txt(self,path):
        with open(path,'r') as f:
            text = f.read().splitlines()
            f.close()
        return text
    def tweet_clean(self,tweet):
        # Remove tickers
        sent_no_tickers = re.sub(r'\$\w*', '', tweet)
        text = sent_no_tickers

        tw_tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        temp_tw_list = tw_tknzr.tokenize(sent_no_tickers)

        # Remove stopwords
        list_no_stopwords = [i for i in temp_tw_list if i.lower() not in cache_english_stopwords]

        # Remove hyperlinks
        list_no_hyperlinks = [re.sub(r'(https)', '', i) for i in list_no_stopwords]

        # Remove hashtags
        list_no_hashtags = [re.sub(r'#', '', i) for i in list_no_hyperlinks]

        # Remove Punctuation and split 's, 't, 've with a space for filter
        list_no_punctuation = [re.sub(r'[' + string.punctuation + ']+', ' ', i) for i in list_no_hashtags]

        # Remove multiple whitespace
        new_sent = ' '.join(list_no_punctuation)
        # Remove any words with 2 or fewer letters
        filtered_list = tw_tknzr.tokenize(new_sent)
        list_filtered = [re.sub(r'^\w\w?$', '', i) for i in filtered_list]
        return [i for i in list_filtered if i!= ""]
    def sentimental_analysis(self,texts, multi = True, core= 4):
        if multi:  ##采用并列计算
            p = Pool(core)
            scores = p.map(self.do_sentiment,texts)
            p.close()
        else:     ##单核循环  <- 小数据可能更快
            scores = []
            for text in texts:
                scores.append(self.do_sentiment(text))
        return scores

    def do_sentiment(self,text):
        print(text)
        if isinstance(text,str)== False:
            return 0
        cleaned_text = self.tweet_clean(text)
        if self.debug:
            print("清理前的文章:")
            print(text)
            print("清理完的文章: ")
            print(cleaned_text)

        if self.prob:  ## 有概率的情况，单词表是 dict 例如 {"great":0.9,"good":0.7，...}
            sent = 0
            for i in cleaned_text:
                if i in self.sen_dict:
                    sent += self.sen_dict[i]
            if sent > 0:
                return 1
            if sent < 0:
                return -1
            else:
                return 0
        else:          ## 单词无概率的情况, 单词表是 list
            pos = 0
            neg = 0
            for i in cleaned_text:
                if i in self.pos_dict:
                    pos += 1
                elif i in self.neg_dict:
                    neg += 1
            if self.debug:
                #print("采取指数大的那一方，正->1 副->1 相同->0")
                print('Positive指数: %.3f' % pos)
                print('Negative指数: %.3f' % neg)
            if pos > neg:
                return 1   #positive comment
            elif pos < neg:
                return -1  #negative comment
            else:
                return 0   #objective comment
    def time_to_date(self,t):
        return datetime.datetime.strptime(t,"%Y-%m-%d %H:%M:%S").date()
    def convert_time_to_date(self,timelist,multi=True):
        if multi:
            p = Pool()
            dates = p.map(self.time_to_date,timelist)
            p.close()
        else:
            dates = []
            for t in timelist:
                dates.append(self.time_to_date(t))
        return dates




if __name__ == "__main__":
    ## 1. 载入推特
    print('正在载入推特文档:', end="\t")
    tweets = pd.read_excel(TEXT)
    print('载入完毕')
    # 清理缺失数据
    tweets = tweets[tweets.isnull().any(axis=1) ==False]
    print(tweets)

    # 2. 找出texts的那一列
    texts = tweets.iloc[:,TEXT_COL]

    # 3. 初始化组 (载入词典， 定义词典类型)
    #s = Sentiment(pos_dict=POS_LIST, neg_dict=NEG_LIST, prob=False, debug=True)    #<- 无概率词典
    s = Sentiment(pos_dict= POS_LIST, neg_dict= NEG_LIST,prob=True,debug= True )   #<- 有概率词典 AFINN-111

    # 4. 做情感分析
    scores = s.sentimental_analysis(texts, multi=False, core = 4)

    # 5. 查看分数（可略过)
    #print(scores)

    # 6. 为每个推特定义时间（秒->天)
    times = tweets.iloc[:,TIME_COL]
    dates = s.convert_time_to_date(times,multi=True)
    print(len(dates))

    # 7. 收集每一天的pos和neg的数量
    pos_score= np.array([0]*len(dates))
    neg_score = np.array([0] * len(dates))
    for i in range(len(dates)):
        if scores[i] == 1:
            pos_score[i] = 1
        elif scores[i] == -1:
            neg_score[i] = 1
    df = pd.DataFrame({"Date":dates,"Pos_score":pos_score,"Neg_score":neg_score})

    #print(df.groupby(lambda x: x.Date).count())
    agg_scores = df.groupby('Date')
    df_s = agg_scores.sum()
    print(df_s)

    ## 画图
    x = df_s.index
    y1 =df_s["Pos_score"]
    y2 =df_s["Neg_score"]
    plt.figure(figsize=(10,5))
    plt.plot(x, y1,"-o",alpha=0.7)
    plt.plot(x, y2,"-x",alpha=0.7)
    plt.grid()
    plt.title('Sentimental Analysis for Trump (AFINN Dict)')
    plt.xlabel('Date')
    plt.ylabel('Comment Freq.')
    plt.legend(["Pos","Neg"])


    ## 抽出推特数量比较多的天数
    twi = 10  # <- 一天推特数大于10就抽取日期
    important_date = []
    print("找出推特数大于%i的所有日期" %twi)
    for i in range(df_s.shape[0]):
        if df_s.iloc[i,:].sum() >  twi:
            important_date.append( (df_s.index[i],df_s.iloc[i,0],df_s.iloc[i,1]) )
    for i in important_date:
        print("%s 推特数有 %i, 其中Pos= %i, Neg= %i" %(i[0],i[1]+i[2],i[1],i[2]))

    plt.show()