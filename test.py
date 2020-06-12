from devnet_kdd19 import load_model_weight_predict
from utils import aucPerformance, dataLoading, prec

model_path = '/Users/roberto/DevNet-master copy 3/model/devnet_/nslkdd_0.0cr_512bs_457ko_4d.h5'
network_depth = 4  # default is 2, if you changed it while training, you should alter it
input_shape = [122]  # [29] is the input shape of `credit card fraud` dataset, if you use other dataset, it may be diff

#x_test, y_test = dataLoading('/Users/roberto/DevNet-master copy 3/dataset/creditcard_test.csv')

y_test = np.load('/Users/roberto/DevNet-master copy 3/dataset/ytest_NSL-KDD.npy')

#y_test = np.load('/Users/roberto/DevNet-master copy 3/dataset/y_test_kddcup.npy')

x_test = np.load('/Users/roberto/DevNet-master copy 3/dataset/testT_NSL-KDD.npy')
#x_test = np.load('/Users/roberto/DevNet-master copy 3/dataset/X_test_kddcup.npy')

scores = load_model_weight_predict(model_path,
                                   input_shape=input_shape,
                                   network_depth=network_depth,
                                   x_test=x_test)

AUC_ROC, AUC_PR = aucPerformance(scores, y_test)

max_prec, max_rec, f1_ = prec(scores, y_test)
