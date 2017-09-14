pp=[]
gtt=[]
import numpy as np
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def mae(predictions, targets):
    return np.absolute(predictions-targets).mean()
for i in range(1,6):
    p = []
    gt = []
    with open("prediction_u{:}.txt".format(i))as f:
        for line in f:
            p.append(float(line.strip()))
    with open("ml-100k/u{:}.test".format(i))as f:
        for line in f:
            gt.append(int(line.split("\t")[2]))
    pp.extend(p)
    gtt.extend(gt)
    print("RMSE for u{:}.test is {:}".format(i,rmse(np.array(p), np.array(gt))))
    print("MAE for u{:}.test is {:}".format(i,mae(np.array(p), np.array(gt))))
    print('----------------------------')
print("avg RMSE for u1~u5 is {:}".format(rmse(np.array(pp), np.array(gtt))))
print("avg MAE for u1~u5 is {:}".format(mae(np.array(pp), np.array(gtt))))
