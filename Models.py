# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import cPickle as cp
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesRegressor as ETR

np.set_printoptions(precision=4, suppress=True)

# <codecell>

#RMSLE - root mean square log error 
def metric(pred, label):
    #return np.sqrt(np.mean((np.log(1 + pred) - np.log(1 + label))**2, axis=0))
    #everything preprocessed with log so we just calc RMSE
    return np.sqrt(np.mean((pred - label)**2, axis=0))

# <codecell>

#RMSLE plus pre-scaling of the data by a polynomial derived from review age, giving 'useful vote density' in time
def poly_metric(_pred, _label, _ages, pa, pb):
    mask = _ages > 40000
    pr = _pred.copy()
    la = _label.copy()
    pr[mask] *= pb(_ages[mask]/1000.)
    pr[~mask] *= pa(_ages[~mask]/1000.) 
    la[mask] *= pb(_ages[mask]/1000.)
    la[~mask] *= pa(_ages[~mask]/1000.)
    return np.sqrt(np.mean((np.log(1 + pr) - np.log(1 + la))**2, axis=0))

# <codecell>

def load_set(model_str):
    ret = {}
    ret['trd'] = cp.load(open('D_TR_'+model_str+'_data'))
    ret['trl'] = cp.load(open('D_TR_'+model_str+'_labels'))
    ret['tra'] = cp.load(open('D_TR_'+model_str+'_ages'))
    ret['ted'] = cp.load(open('D_TE_'+model_str+'_data'))
    ret['tei'] = cp.load(open('D_TE_'+model_str+'_ids'))
    ret['tea'] = cp.load(open('D_TE_'+model_str+'_ages'))
    ret['cut'] = int(len(ret['trd']) * 0.9)
    #ret.update(cp.load(open('D_label_poly')))
    print model_str, 'set | train entries:', ret['trd'].shape, 'cut:', ret['cut'], 'Test:', ret['ted'].shape
    return ret

# <codecell>

def fit_pred(_models, _data, _labels, _vcut, _ages=None, pa=None, pb=None):
    pred_sum = 0
    for _model in _models.values():
        model_name = str(_model).split('(')[0]
        _model.fit(_data[:_vcut], _labels[:_vcut])
        _preds = _model.predict(_data[_vcut:])
        print model_name, metric(_preds, _labels[_vcut:])
        #print model_name, poly_metric(_preds, _labels[_vcut:], _ages[_vcut:], pa, pb)
        pred_sum += _preds
    if(len(_models.values()) > 1): print '@@Together: ', metric(pred_sum/len(_models), _labels[_vcut:])

# <codecell>

#Define models
n_est = 20

all_models = {'RFR': RFR(n_estimators=n_est, n_jobs=4),
              'ETR': ETR(n_estimators=n_est, n_jobs=4)}
user_models = {'RFR': RFR(n_estimators=n_est, n_jobs=4),
              'ETR': ETR(n_estimators=n_est, n_jobs=4)}
votes_models = {'RFR': RFR(n_estimators=n_est, n_jobs=4),
              'ETR': ETR(n_estimators=n_est, n_jobs=4)}

# <codecell>

#Load sets of data
all_set = load_set('all')
votes_set = load_set('votes')
user_set = load_set('user')

# <codecell>

#Fit all of the models and get individual and composite scores
%%time
fit_pred(all_models, all_set['trd'], all_set['trl'], all_set['cut'])
fit_pred(votes_models, votes_set['trd'], votes_set['trl'], votes_set['cut'])
fit_pred(user_models, user_set['trd'], user_set['trl'], user_set['cut'])

# <codecell>

#Check how important each feature is
print '###All user info model###'
for key in all_models:
    print key, all_models[key].feature_importances_
print '###Missing user model###'
for key in user_models:
    print key, user_models[key].feature_importances_
print '###Missing useful votes model###'
for key in votes_models:
    print key, votes_models[key].feature_importances_

# <codecell>

test_preds_all = np.expm1(np.mean([_model.predict(all_set['ted']) for _model in all_models.values()], axis=0))
test_preds_user = np.expm1(np.mean([_model.predict(user_set['ted']) for _model in user_models.values()], axis=0))
test_preds_votes = np.expm1(np.mean([_model.predict(votes_set['ted']) for _model in votes_models.values()], axis=0))

# <codecell>

test_preds_all[test_preds_all < 0] = 0.
test_preds_user[test_preds_user < 0] = 0.
test_preds_votes[test_preds_votes < 0] = 0.

# <codecell>

#Prepare submission
f = open('submission', 'w')
f.write('Id,Votes\n')
for i in range(len(all_set['tei'])):
    f.write(all_set['tei'][i]+','+str(test_preds_all[i])+'\n')
for i in range(len(user_set['tei'])):
    f.write(user_set['tei'][i]+','+str(test_preds_user[i])+'\n')
for i in range(len(votes_set['tei'])):
    f.write(votes_set['tei'][i]+','+str(test_preds_votes[i])+'\n')
f.close()

