import numpy as np

def L1(yhat,y):
    loss=sum(abs(y-yhat))
    return loss

def L2(yhat,y):
    loss=np.dot(y-yhat,yhat-y)
    return loss



yhat=np.array([.9,0.2,0.1,0.4,0.9])
y=np.array([1,0,0,1,1])
print("L1="+str(L1(yhat,y)))
print("L2="+str(L2(yhat,y)))
print("L2-L1="+str((y-yhat)))


