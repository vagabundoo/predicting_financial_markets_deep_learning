import pandas as pd
import numpy as np

test = np.array(
    [[1,2,3],
    [4,5,6]
    ])

def toTimeseries(df, periodInput):
    # X= []
    # y= []
    # for i in range(len(df) - periodInput -1):
    #     t = [df[[i+j], : ]] for j in range(0, periodInput)]
    #     return t
    #     x.append(t)
    #     #t.append()
    X=[]
    y=[]
    for i in range(len(df)-periodInput-1):
        t=[]
        for j in range(0,periodInput):
            t.append(df[[(i+j)], :])
        X.append(t)
        y.append(df[i+ periodInput,0])
    return(X,y)

print(toTimeseries(test, 2))
