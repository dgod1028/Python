import pandas as pd
import gensim as gs
from gensim import corpora
from datetime import  datetime
import re
import time
from nltk.corpus import stopwords
import math
import string

def main():
    start = time.time()
    """
    sp = 1800
    doc1 = pd.read_csv("d:/study/paper/iPhone_comment.csv").sample(sp)
    doc2 = pd.read_csv("d:/study/paper/iPhone3G_comment2.csv").sample(1000)
    doc3 = pd.read_csv("d:/study/paper/iPhone3GS_comment.csv").sample(sp)
    doc4 = pd.read_csv("d:/study/paper/iPhone4_comment.csv").sample(sp)
    doc5 = pd.read_csv("d:/study/paper/iPhone4S_comment.csv").sample(sp)
    doc6 = pd.read_csv("d:/study/paper/iPhone5_comment.csv").sample(sp)
    doc7 = pd.read_csv("d:/study/paper/iPhone5C_comment.csv").sample(sp)
    doc8 = pd.read_csv("d:/study/paper/iPhone5S_comment.csv").sample(sp)
    #print(doc.iloc[:,2])
    doc =pd.DataFrame(pd.concat([doc1,doc2,doc3,doc4,doc5,doc6,doc7,doc8],axis=0))

    doc.to_csv("d:/study/paper/sample.csv",index=False,encoding="utf-8")
    print(doc +1)
 """
    doc = pd.read_csv("d:/study/paper/sample.csv")
    doc = doc.dropna()
    print(doc.shape)
    #print(doc2)
    time1 = []
    corpus = []

    #print(doc)
    #doc = pd.DataFrame(doc.iloc[0:49, :])
    for i in range(doc.shape[0]):
        try:
            temp = datetime.strptime(doc.iloc[i, 2], "%d %b %Y")
        except:
            temp = datetime.strptime("30 May 2017","%d %b %Y")
        time1.append(datetime.strftime(temp,"%Y-%m") )
        doc.iloc[i,2] = datetime.strftime(temp,"%Y-%m")
    for i in range(len(time1)):
        corpus.append(doc.iloc[i,1])
    doc = doc.sort(columns="date")


    corpus.reverse()
    time1.reverse()
    time2 = []
    t = []
    temp = "00"
    count = 0
    timeslice = []
    for i in range(doc.shape[0]):
        time2.append(re.split("-", doc.iloc[i,2]))
        if temp == time2[i][0]:
            t.append(count)
        else:
            count += 1
            temp = time2[i][0]
            t.append(count)
    print (t)
    #print(len(t))
    #print(doc)
    #print(time2)
    #print(doc.iloc[-2,1])
    temp = 1
    count = 0
    for i in t :
        if i == temp:
            count += 1
        else:
            timeslice.append(count)
            count = 1
            temp = i
    timeslice.append(count)
    #print("## TimeSlice")
    #print(timeslice)
    for i in range(len(corpus)):
        #print("##",i)
        #print(corpus[i])
        corpus[i] = re.sub("[^\w\s]","",corpus[i])

    #stoplist = set('for a of the and to in i it is this that but i am you are my your me can if no not in on or will be was as by at its than have has with im do does so'.split())
    stoplist = set(stopwords.words("english"))
    print(stoplist)
    texts = [[word for word in document.lower().split() if word not in stoplist]
                   for document in corpus]
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
           frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 3]
            for text in texts]
    #print(texts)
    dict = corpora.Dictionary(texts)
    dict.save("dictionary6.dict")
    #print(dict)
    doc = [dict.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize("doc6.mm",doc)
    #print(doc)
    model =gs.models.wrappers.DtmModel("dtm-master/bin/dtm-win64.exe",doc,time_slices=timeslice,
                                       alpha=0.1,top_chain_var=0.01,
                                       num_topics=5,id2word=dict,lda_max_em_iter=50,
                                       rng_seed=1)
    model.save("model6.dtm")
    print(model.print_topics(2,5,10))
    print(model.show_topics(-1))
    print(model.print_topic(1,1))
    print(time.time() -start )
    pd.DataFrame(t).to_csv("d:/study/paper/t.csv")
if __name__ == "__main__":
    main()
    ### 1   a = 0.5 beta = 0.3 seed = 1
    ### 2   a = 1 beta = 1 seed = 1
    ### 3   a = 1 beta = 1 seed = 2
    ### 4   a = 1 beta = 1 seed = 3
    ### 5   a = 0.1 beta = 0.1 seed = 1
    ### 6   a = 0.1 beta = 0.1 seed = 2
    ### 7   a = 0.1 beta = 0.01 seed = 1
