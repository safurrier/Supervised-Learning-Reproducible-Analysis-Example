
# coding: utf-8

# In[84]:


import ruamel.yaml as yaml
import os
import sys
import pandas as pd
import numpy as np

NO_CONFIG_ERR_MSG = """No config file found. Root directory is determined by presence of "config.yaml" file."""

original_wd = os.getcwd()

# Number of times to move back in directory
num_retries = 10
for x in range(0, num_retries):
    # try to load config file
    try:
        with open("config.yaml", 'r') as stream:
            cfg = yaml.safe_load(stream)
    # If not found move back one directory level
    except FileNotFoundError:
        os.chdir('../')
        # If reached the max number of directory levels change to original wd and print error msg
        if x+1 == num_retries:
            os.chdir(original_wd)
            print(NO_CONFIG_ERR_MSG)

# Add directory to PATH
path = os.getcwd()

if path not in sys.path:
    sys.path.append(path)


# # Load Madelon data and concat together

# In[85]:


madelon_train = pd.read_csv('data/raw/madelon/madelon_train.data.txt', header=None, sep=' ')
madelon_test = pd.read_csv('data/raw/madelon/madelon_valid.data.txt', header=None, sep=' ')
madelon_data = pd.concat([madelon_train, madelon_test], 0).astype(float)
madelon_train_labels = pd.read_csv('data/raw/madelon/madelon_train.labels.txt',header=None,sep=' ')
madelon_test_labels = pd.read_csv('data/raw/madelon/madelon_valid.labels.txt',header=None,sep=' ')
madelon_labels = pd.concat([madelon_train_labels, madelon_test_labels], 0)
madelon_labels.columns = ['Class']
madelon_df = pd.concat([madelon_data, madelon_labels],1)
madelon_df = madelon_df.dropna(axis=1,how='all')
madelon_df.to_hdf('data/processed/datasets.hdf', 'madelon', complib='blosc', complevel=9)


# In[86]:


madelon_df = pd.read_hdf('data/processed/datasets.hdf', key='madelon')
madelon_df.head()


# ## Load Car Data and Clean

# In[127]:


# cars_df = pd.read_csv('data/raw/cars/car.data.txt', header=None, 
#                       names=[
# "buying ", 
# "maint", 
# "doors", 
# "persons", 
# "lug_boot", 
# "safety",
#                           'class'
#                       ])
# cars_df.head()


# In[88]:


# # Changing to binary classification problem
# # Acceptable, Good and Very Good all become the positive class 1
# # Unacceptable is the negative class 0
# cars_df['Class'] = cars_df['Class'].replace({'unacc':0,'acc':1,'vgood':2,'good':2})


# In[89]:


# cars_df['doors'] = cars_df['doors'].replace({'5more':5}).apply(pd.to_numeric)


# In[90]:


# one_hot_columns = pd.get_dummies(cars_df.select_dtypes(include='object')).rename(columns=lambda x: x.replace('-','_'))
# cars_df = pd.concat([one_hot_columns, cars_df[['doors','class']]], axis=1)


# In[124]:


# cars_df.to_hdf('data/processed/datasets.hdf','cars',complib='blosc',complevel=9)


# ## Load Income Data

# In[92]:


# # Preprocess with adult dataset
# adult = pd.read_csv('data/raw/census-income/adult.data.txt',header=None)
# adult.columns = ['age','employer','fnlwt','edu','edu_num','marital','occupation','relationship','race','sex','cap_gain','cap_loss','hrs','country','income']
# # Note that cap_gain > 0 => cap_loss = 0 and vice versa. Combine variables.
# adult['cap_gain_loss'] = adult['cap_gain']-adult['cap_loss']
# adult = adult.drop(['fnlwt','edu','cap_gain','cap_loss'],1)
# adult['income'] = pd.get_dummies(adult.income)

# # Aggregate Countries to Higher Level Grouping
# #http://scg.sdsu.edu/dataset-adult_r/
# replacements = { 'Cambodia':' SE-Asia',
#                 'Canada':' British-Commonwealth',
#                 'China':' China',
#                 'Columbia':' South-America',
#                 'Cuba':' Other',
#                 'Dominican-Republic':' Latin-America',
#                 'Ecuador':' South-America',
#                 'El-Salvador':' South-America ',
#                 'England':' British-Commonwealth',
#                 'France':' Euro_1',
#                 'Germany':' Euro_1',
#                 'Greece':' Euro_2',
#                 'Guatemala':' Latin-America',
#                 'Haiti':' Latin-America',
#                 'Holand-Netherlands':' Euro_1',
#                 'Honduras':' Latin-America',
#                 'Hong':' China',
#                 'Hungary':' Euro_2',
#                 'India':' British-Commonwealth',
#                 'Iran':' Other',
#                 'Ireland':' British-Commonwealth',
#                 'Italy':' Euro_1',
#                 'Jamaica':' Latin-America',
#                 'Japan':' Other',
#                 'Laos':' SE-Asia',
#                 'Mexico':' Latin-America',
#                 'Nicaragua':' Latin-America',
#                 'Outlying-US(Guam-USVI-etc)':' Latin-America',
#                 'Peru':' South-America',
#                 'Philippines':' SE-Asia',
#                 'Poland':' Euro_2',
#                 'Portugal':' Euro_2',
#                 'Puerto-Rico':' Latin-America',
#                 'Scotland':' British-Commonwealth',
#                 'South':' Euro_2',
#                 'Taiwan':' China',
#                 'Thailand':' SE-Asia',
#                 'Trinadad&Tobago':' Latin-America',
#                 'United-States':' United-States',
#                 'Vietnam':' SE-Asia',
#                 'Yugoslavia':' Euro_2'}
# # Strip whitespace
# adult['country'] = adult['country'].str.strip()

# # Replace Countries, unemployment in employer column, and combine Husband and Wife to Spouse
# adult = adult.replace(to_replace={'country':replacements,
#                                   'employer':{' Without-pay': ' Never-worked'},
#                                   'relationship':{' Husband': 'Spouse',' Wife':'Spouse'}})    
# adult['country'] = adult['country'].str.strip()

# # Strip whitespace in string columns
# for col in ['employer','marital','occupation','relationship','race','sex','country']:
#     adult[col] = adult[col].str.strip()
    
# # One hot encode data and rename columns to underscores to allow for 
# # Pandas column accessors
# adult = pd.get_dummies(adult)
# adult = adult.rename(columns=lambda x: x.replace('-','_'))

# adult.to_hdf('data/processed/datasets.hdf','adult',complib='blosc',complevel=9)


# ## Load Abalone Data

# In[123]:


column_names = ["Sex",
"Length",
"Diameter",
"Height",
"Whole weight",	
"Shucked weight",
"Viscera weight",
"Shell weight",
"Rings"]

abalone_df = pd.read_csv('data/raw/abalone/abalone.txt', names=column_names)

# Create classification problem based on Rings, which are a function of age (Age = 1.5 + Rings)
#abalone_df['Class'] = np.where(abalone_df['Rings'] > 9, 1, 0) Binary Classification

# Multilabel classification
# Infants, Most prevalent age (9-10) before harvesting
# Full adults
abalone_df['Class'] = pd.cut(abalone_df['Rings'], bins=[-1, 8.9, 10.1, 100], labels=[0, 1, 2])

# One hot encode gender and concat together
abalone_df = pd.concat([pd.get_dummies(abalone_df['Sex'], prefix='Sex'), 
                       abalone_df.drop(columns=['Sex', 'Rings'], axis=1)], axis=1)

# Export data
abalone_df.to_hdf('data/processed/datasets.hdf','abalone', format='table', complib='blosc',complevel=9)
