from keras.models import load_model, Model
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn.metrics import roc_curve,confusion_matrix,f1_score, roc_auc_score

# Convert comma separated to list of values
def cat(data,field,leng,s=','):
  data[field] = data[field].map(lambda x:x.split(s))
  data[leng] = data[field].map(lambda x:len(x))

# Helper Function
def softmax(x,axis=1):
  ndim = K.ndim(x)
  if ndim==2:
    return K.softmax(x)
  elif ndim >2:
    e = K.exp(x - K.max(x, axis=axis, keepdims=True))
    s = K.sum(e, axis=axis, keepdims =True)
    return e/s
  else:
    raise ValueError('Cannot apply softmax to a tensor that is 1D')


# Validation plot to improve model performance
def plot_loss(history,p_epochs):
  ylims = range(1,p_epochs+1,10)
  plt.plot(history.history['loss'],color='red',label='train loss')
  plt.xticks(ylims)
  plt.legend(loc=1)
  plt.title('train loss vs epochs')

def plot_acc(history,p_epochs):
  ylims = range(1,p_epochs+1,10)
  plt.plot(history.history['accuracy'],label='accuracy',c='r')    
  plt.xticks(ylims)
  plt.legend(loc=4)
  plt.title('train acc vs epochs')


# Model Performance
def auc_score_train(threshold,model,myDict):
  prob = model.predict([myDict['Cust_Path_Train'],myDict['s0_tr'],myDict['Cust_Demo_Train']])
  cl = [1 if p > threshold else 0 for p in prob]
  print(confusion_matrix(myDict['Enrolled_Train'],cl))
  print(metric(myDict['Enrolled_Train'],prob,cl,label='train dataset performance'))

def auc_score_test(threshold,model,myDict):
  prob = model.predict([myDict['Cust_Path_Test'],myDict['s0_te'],myDict['Cust_Demo_Test']])
  cl = [1 if p > threshold else 0 for p in prob]
  print(confusion_matrix(myDict['Enrolled_Test'],cl))
  print(metric(myDict['Enrolled_Test'],prob,cl,label='test dataset performance'))
  
def metric(y_valid,prob,cl,label=None):
  fpr, tpr, threshold = roc_curve(y_valid, prob)
  auc = roc_auc_score(y_valid,prob)
  plot_roc_curve(fpr,tpr,label=label)
  acc = (y_valid==cl).mean()
  print('Accuracy: {:.3f}, AUC: {:.3f}'.format(acc,auc))

def plot_roc_curve(fpr, tpr, label=None):
  plt.plot(fpr, tpr, linewidth=2, label=label)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.axis([0, 1, 0, 1])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')


# Save model trained weights
def save_weight(name,model):
  model.save_weights(name)


# Load Saved Weights
def load_weight(name,model):
  model.load_weights(name)


