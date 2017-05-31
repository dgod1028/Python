import pandas as pd
import gensim as gs
from gensim import corpora
import numpy as np
from datetime import  datetime
import re

import time

def main():
    model = gs.models.wrappers.DtmModel.load("model.dtm")
    doc = corpora.MmCorpus("doc.mm")
    dat = []
    for i in range(11):
            dat.append(model.dtm_vis(doc,i))
    docdis = pd.DataFrame(pd.concat((pd.DataFrame(x[0]) for x in dat),axis=1 ))


    #print(model.print_topics(2,5,10))
    #print(model.show_topics(10,5,10))
    #print(len(model.show_topics(10,5,10) )  )
    #print(model.print_topic(1,1))
    #print(model.dtm_vis())
    top = model.show_topics(5,11,10)

    print(len(top))
    for i in range(len(top)):
        top[i] = re.split(" \+ ",top[i])
        #print(top[i])
        for l in range(len(top[i])):
            top[i][l] = re.split("\*", top[i][l])
        #print(top[i])
    topwords = pd.DataFrame(pd.concat((pd.DataFrame(x) for x in top),axis=1 ))
    print(topwords)
    topwords.to_csv("d:/study/paper/topwords.csv")
    docdis.to_csv("d:/study/paper/docdis.csv")
if __name__ == "__main__":
    main()
