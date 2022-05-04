import numpy as np

def f(a):
    return '%.4f\t %.4f\t'% (np.mean(a), np.std(a))


# mlc bert from partial logs
f1 = [0.7311, 0.7224, 0.7050, 0.6865, 0.6761]
p  = [0.7174, 0.7081, 0.7051, 0.6822, 0.6587]
r  = [0.7453, 0.7372, 0.7050, 0.6908, 0.6944]
print ('BERT', f(p), f(r), f(f1))

# mlc roberta from partial logs
f1 = [0.6743, 0.7004, 0.6654, 0.6887, 0.6867] 
p  = [0.6610, 0.6764, 0.6511, 0.6769, 0.6735]
r  = [0.6881, 0.7261, 0.6804, 0.7009, 0.7005]
print ('RoBERTa', f(p), f(r), f(f1))
