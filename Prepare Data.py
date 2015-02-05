# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os, time, math, copy, re, string
import numpy as np
import cPickle as cp
from nltk import PorterStemmer as PS
from collections import defaultdict
import matplotlib.pyplot as plt
from y_util import read_json, age_in_hours

# <headingcell level=3>

# Load data

# <codecell>

tr_path = os.getcwd() + "/yelp_training_set/"
te_path = os.getcwd() + "/yelp_test_set/"
tr_base = "yelp_training_set_"
te_base = "yelp_test_set_"
ext = ".json"

# <codecell>

#Get training data from files
tr_bus = read_json(tr_path+tr_base+'business'+ext)
tr_chk = read_json(tr_path+tr_base+'checkin'+ext)
tr_rev = read_json(tr_path+tr_base+'review'+ext)
tr_usr = read_json(tr_path+tr_base+'user'+ext)

# <codecell>

#Get testing data from files
te_bus = read_json(te_path+te_base+'business'+ext)
te_chk = read_json(te_path+te_base+'checkin'+ext)
te_rev = read_json(te_path+te_base+'review'+ext)
te_usr = read_json(te_path+te_base+'user'+ext)

# <headingcell level=3>

# Create user and business dictionaries

# <codecell>

#Setup dictionaries for users and businesses
usr_dict = {entry['user_id']: entry for entry in (tr_usr + te_usr)}
bus_dict = {}
for entry in tr_bus:
    bus_dict[entry['business_id']] = entry
    bus_dict[entry['business_id']]['checkins'] = 0
for entry in te_bus:
    bus_dict[entry['business_id']] = entry
    bus_dict[entry['business_id']]['checkins'] = 0
#Use checkins as a bulk number
for entry in tr_chk:
    bus_dict[entry['business_id']]['checkins'] = sum(entry['checkin_info'].values())

# <headingcell level=3>

# Create features from raw data + dictionaries

# <codecell>

#regex to remove punctuation
regex_punc = re.compile('[%s]' % re.escape(string.punctuation))

def make_feats(dataset, ref_date, _bus_dict, _usr_dict, test=False):
    #We generate 3 datasets depending on available information: all, user votes missing, or the entire user unknown
    no_user = {}
    no_votes = {}
    all_info = {}
    
    for sample in dataset:
        #Add basic review features
        new_sample = [age_in_hours(sample['date'], ref_date), sample['stars']]
        
        #Add features based on text
        new_sample += get_text_features(sample['text'], thresh=3)
        
        #Grab info from business dict: star rating, rev. ct, and checkins might measure popularity
        #extremeness of user rating might signal special situation
        bus = _bus_dict[sample['business_id']]
        new_sample += [bus['stars'], (sample['stars'] - bus['stars'])**2, 
                       math.log(1.+bus['review_count']), math.log(1.+bus['checkins'])]
        
        #Get info and split set based on user dict
        user = sample['user_id']
        if not test or user not in _usr_dict:
            no_user[sample['review_id']] = copy.copy(new_sample)
            
        #We'll use the log(# of reviews), average star rating of user, and how extreme the sample rating is
        if user in _usr_dict:
            entry = _usr_dict[user]
            u_rev_ct = entry['review_count']
            new_sample += [math.log(1.+u_rev_ct), entry['average_stars']]
            new_sample += [(sample['stars'] - entry['average_stars'])**2]
            
            if not test or 'votes' not in entry:
                no_votes[sample['review_id']] = copy.copy(new_sample)
                
            #Vote history of this user's reviews is useful; use as a density
            if 'votes' in entry:
                new_sample += [entry['votes']['cool']*1./u_rev_ct,
                               entry['votes']['funny']*1./u_rev_ct,
                               entry['votes']['useful']*1./u_rev_ct]
                            
                all_info[sample['review_id']] = new_sample

    return no_user, no_votes, all_info

# <codecell>

#Grabs a few simple features like text length, how much punctuation, fully capitalized words, unique words, avg word length
def get_text_features(_text, thresh=3):
    text_len = len(_text)
    proc = regex_punc.sub('', _text)
    n_punc = text_len - len(proc) 
    n_cap, hits = count_words(proc)
    unique_words = len(hits)
    avg_len = text_len * 1. / (1. + sum(hits.values()))
    return [math.log(1.+text_len), n_punc, n_cap, unique_words, avg_len]

# <codecell>

def count_words(_text, thresh=3):
    _hits = defaultdict(int)
    _ps = PS()
    ct = 0
    for word in _text.split():
        if word.upper() == word and len(word) >= thresh:
            ct += 1
        _hits[_ps.stem(word.lower())] += 1
    return ct, _hits

# <codecell>

#Rescale the labels by a polynomial based on useful_votes / review age data
# --two quadratics are fit to the scatter plot of useful votes vs review age (data shows non-trivial dependence on age)
# --One could maybe motivate a Gumbel/Frechet/Weibull distribution or some such, but that's a bit involved...
def make_density_labels(_data, _ages):
    ct = np.zeros(70)
    val = np.zeros(70)
    #fit the avg_useful/age data to 2 quadratics (see analysis below)
    ###Some bias due to binning###
    for i in _data:
        rev_age = _ages[i['review_id']]
        ct[rev_age//1000] += 1
        val[rev_age//1000] += i['votes']['useful']
    val /= ct
    ct /= np.sum(ct)
    #For whatever reason, there is a peak at ~40k hours, after which the usefulness with age decreases
    pa = np.poly1d(np.polyfit(np.arange(40), val[:40], 2))
    pb = np.poly1d(np.polyfit(np.arange(40,70), val[40:], 2))
    ret = {}
    _ages = {}
    for i in _data:
        rev_age = _ages[i['review_id']]
        ret[i['review_id']] = i['votes']['useful']
        _ages[i['review_id']] = rev_age
        if rev_age > 40000:
            ret[i['review_id']] /= pb(rev_age/1000.)
        else:
            ret[i['review_id']] /= pa(rev_age/1000.)
    return ret, _ages, pa, pb

# <codecell>

%%time
#Create train and test data
tr_no_user, tr_no_votes, tr_all = make_feats(tr_rev, '2013-01-19', bus_dict, usr_dict, test=False)
te_no_user, te_no_votes, te_all = make_feats(te_rev, '2013-03-26', bus_dict, usr_dict, test=True)

#Create review ages and labels (log)
ages = {i['review_id']: age_in_hours(i['date'], '2013-01-19') for i in tr_rev}
ages.update({i['review_id']: age_in_hours(i['date'], '2013-03-26') for i in te_rev})
labels = {i['review_id']: math.log(1.+i['votes']['useful']) for i in tr_rev}
#labels, pa, pb = make_density_labels(tr_rev, ages)

print len(tr_no_user), len(tr_no_votes), len(tr_all)
print len(te_no_user), len(te_no_votes), len(te_all)

# <codecell>

def prep_datafiles(_data, _labels, _ages, fbase, test=False):
    out_ids = []
    out_data = []
    out_ages = []
    out_labels = []
    for _id in _data:
        out_ids.append(_id)
        out_data.append(_data[_id])
        out_ages.append(_ages[_id])
        if not test:
            out_labels.append(_labels[_id])
    out_data = np.asarray(out_data, dtype=np.float32)
    out_labels = np.asarray(out_labels, dtype=np.float32)
    out_ages = np.asarray(out_ages, dtype=np.float32)
    cp.dump(out_ids, open(fbase+'_ids', 'wb'), 2)
    cp.dump(out_data, open(fbase+'_data', 'wb'), 2)
    cp.dump(out_ages, open(fbase+'_ages', 'wb'), 2)
    if not test:
        cp.dump(out_labels, open(fbase+'_labels', 'wb'), 2)

# <codecell>

#Dump all of the datasets for easy use with the Models notebook
prep_datafiles(tr_no_user, labels, ages, 'D_TR_user')
prep_datafiles(tr_no_votes, labels, ages, 'D_TR_votes')
prep_datafiles(tr_all, labels, ages, 'D_TR_all')
prep_datafiles(te_no_user, labels, ages, 'D_TE_user', test=True)
prep_datafiles(te_no_votes, labels, ages, 'D_TE_votes', test=True)
prep_datafiles(te_all, labels, ages, 'D_TE_all', test=True)
#cp.dump({'pa': pa, 'pb': pb}, open('D_label_poly', 'wb'), 2)

# <headingcell level=3>

# Analysis

# <codecell>

%matplotlib inline

# <codecell>

#Plot histograms of different features of the data, by choosing an index
ind = 0
tr_data = [i[ind] for i in tr_all.values()]
te_data = [i[ind] for i in te_all.values()]
hist, bins = np.histogram(tr_data, bins=50)
hist2, bins2 = np.histogram(te_data, bins=50)
#hist, bins = np.histogram(np.log(1 + tr_data[:, ind]), bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.bar(center, hist2, align='center', width=width, color='r')
plt.show()

# <codecell>

normal_labels = {i['review_id']: i['votes']['useful'] for i in tr_rev}
ct = np.zeros(70)
val = np.zeros(70)
for i in tr_all:
    ct[tr_all[i][0]//1000] += 1
    val[tr_all[i][0]//1000] += normal_labels[i]
val /= ct
ct /= np.sum(ct)
pa = np.poly1d(np.polyfit(np.arange(40), val[:40], 2))
pb = np.poly1d(np.polyfit(np.arange(40,70), val[40:], 2))
xp = np.arange(70)
plt.scatter(xp, val)
plt.plot(xp[:40], pa(xp[:40]))
plt.plot(xp[40:], pb(xp[40:]))
plt.bar(xp, ct*10)
plt.show()

# <codecell>

#Count how many of each type of info situation we have: all info ('votes'), missing user, or just missing vote counts
def c_votes(_str, _data, _dict):  
    counts = defaultdict(int)
    ret = []
    for sample in _data:
        if sample['user_id'] in _dict:
            if _str in _dict[sample['user_id']]:
                counts[_str] += 1
            else:
                counts['no_votes'] += 1
                ret.append(_dict[sample['user_id']])
        else:
            counts['no user'] += 1
    print counts.items()
    return ret

# <codecell>

nv = c_votes('votes', tr_rev, usr_dict)
nv2 = c_votes('votes', te_rev, usr_dict)

