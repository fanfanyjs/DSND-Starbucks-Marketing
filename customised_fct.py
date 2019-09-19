import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
RandomForestRegressor, GradientBoostingRegressor)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, make_scorer, 
r2_score, mean_squared_error)
from xgboost import XGBRegressor, XGBClassifier
import joblib
import pickle
import glob

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def read_data(df):
    
    '''
    Print out shape, data types, number of nulls for each
    '''   
    num_entries = len(df)
    num_col = df.shape[1]
    col_type = df.dtypes
    col_null = df.isnull().sum()
    
    print('There are '+str(num_entries) +' entries')
    print('There are '+str(num_col) +' columns')
    print(' ')
    print('These are the data types --')
    print(col_type)
    print(' ')
    print('The columns have the following number of nulls')
    print(col_null)
    print(' ')
    print(' ')
    
    return df.head()

class RawPreprocess:

    def __init__(self):
        pass
    
    def portfoliodf_trans(self, portfolio):
        '''
        Cleans input portfolio data
        '''

        portfolio_df = portfolio.copy()
        portfolio_df['duration_h'] = portfolio_df['duration']*24

        chan_list = ['web', 'email', 'mobile', 'social']
        for val in chan_list:
            portfolio_df[val] = portfolio_df['channels'].apply(lambda x:1 if val in x else 0)

        portfolio_new = portfolio_df.drop(['channels', 'duration'], axis = 1)

        return portfolio_new

    def profiledf_clean(self, profile):
        '''
        Clean profile data - flag null entries, imputation, one-hot encoding, convert date-time column
        '''

        profile_df = profile.copy()
        profile_df['null_info'] = profile_df['age'].apply(lambda x:1 if x==118 else 0)

        age_mean = profile_df[profile_df['age']!=118]['age'].mean()
        income_mean = profile_df[~np.isnan(profile_df['income'])]['income'].mean()
        profile_df['age'] = profile_df['age'].apply(lambda x:age_mean if x==118 else x)
        profile_df['income'] = profile_df['income'].fillna(income_mean)

        profile_df['join_date'] = pd.to_datetime(profile_df['became_member_on'], format='%Y%m%d')
        latest_date = profile_df['join_date'].max()
        profile_df['days_joined'] = (latest_date - profile_df['join_date']).apply(lambda x:x.days)

        profile_new = pd.get_dummies(profile_df, columns = ['gender'])

        profile_new = profile_new.drop(['became_member_on', 'join_date'], axis = 1)

        return profile_new
    
    def parse_order_id(self, df):
        '''
        Parse value column in transcript dataframe that is a dictionary containing both offer id and transaction amounts
        '''
        value_parse_col = ['offer id', 'offer_id', 'amount']

        for col in value_parse_col:
            df[col] = df['value'].apply(lambda x:x[col] if col in x.keys() else '')

        # Combine the offer id columns
        df['clean_offer_id'] = df["offer_id"].map(str) + df["offer id"]
        df = df.drop(['offer id', 'offer_id', 'value'], axis = 1)

        return df
    
    def transpose_transcript(self, transcript):
        '''
        Transpose transcript data such that each row represent a unique offer sent to a specific customer, 
        specifying whether the offer has been viewed, completed and the transaction amount associated
        '''

        transcript_df = transcript.copy()

        # Step 1: Split transcript data into four dataframes
        offer_col = ['person', 'clean_offer_id', 'time']
        transact_col = ['person', 'amount', 'time']
        portfolio_col = ['difficulty', 'offer_type', 'reward', 'duration_h', 'web', 'email', 'mobile', 'social']

        transcript_received = transcript_df[transcript_df['event']=='offer received'][offer_col + portfolio_col]
        transcript_viewed = transcript_df[transcript_df['event']=='offer viewed'][offer_col]
        transcript_completed = transcript_df[transcript_df['event']=='offer completed'][offer_col]
        transcript_transact = transcript_df[transcript_df['event']=='transaction'][transact_col]


        # Step 2: Join received data with viewed data
        transcript_view = transcript_received.merge(transcript_viewed, how = 'outer', on = ['clean_offer_id', 'person'], \
                              suffixes = ('_rec', '_view'))

        print('View data joined!')

        view_dup = (transcript_view.loc[~np.isnan(transcript_view['time_view'])]\
                   .duplicated(subset = ['person', 'clean_offer_id', 'time_view'], keep = 'first')).sum()

        print('Number of duplicated time_view rows:', view_dup)

        # remove duplicates where viewing time is earlier than receipt time, 
        # and only consider the latest viewed if the same timestamp is duplicated for offers received at different times
        transcript_view.loc[(transcript_view['time_view'] < transcript_view['time_rec']), 'time_view'] = float('NaN')

        transcript_view.loc[:,'view_diff'] = transcript_view['time_view'] - transcript_view['time_rec']
        transcript_view.loc[:,'view_diff_min'] = transcript_view.groupby(['person', 'clean_offer_id', 'time_view']) \
                                                ['view_diff'].transform(min)
        transcript_view.loc[(~np.isnan(transcript_view['time_view'])) & 
                            (transcript_view['view_diff_min']!=transcript_view['view_diff']), 'time_view'] \
                            = float('NaN') 
        transcript_view = transcript_view.drop(['view_diff', 'view_diff_min'], axis = 1)

        transcript_view = transcript_view.sort_values(['time_view']) \
                          .drop_duplicates(subset = ['person', 'clean_offer_id', 'time_rec'], keep = 'first')

        assert (~np.isnan(transcript_view['time_rec'])).sum() == len(transcript_received), 'Duplicate received timestamps'
        assert (~np.isnan(transcript_view['time_view'])).sum() == len(transcript_viewed), 'Duplicate viewed timestamps'
        print('Duplicated time_view eliminated!')
        print(' ')


        # Step 3: Join in completed timestamps
        # As a unique user can complete two of the same type of offer received at different times with 
        # just one transaction (one completion entry can have multiple offer matches), 
        # only the latest viewed offer that is completed within the effective period will be considered 
        # completed in this transformation.

        transcript_comp = transcript_view.copy()
        transcript_comp['effect_timeout'] = transcript_comp['time_rec'] + transcript_comp['duration_h']
        transcript_comp = transcript_comp.sort_values(by = ['time_view', 'time_rec'], ascending = False)
        transcript_comp['time_comp'] = float('NaN')

        for index, row in transcript_completed.drop_duplicates().iterrows():

            user = row['person']
            offer_id = row['clean_offer_id']
            time_comp = row['time']

            criteria = ((transcript_comp['person'] == user) & 
                       (transcript_comp['clean_offer_id'] == offer_id) &
                       (transcript_comp['time_rec'] <= time_comp) &
                       (transcript_comp['time_view'] <= time_comp) &
                       (transcript_comp['effect_timeout'] >= time_comp) &
                       (np.isnan(transcript_comp['time_comp'])))

            trans_idx = transcript_comp.loc[criteria].index

            if len(trans_idx)>0:
                transcript_comp.loc[trans_idx[0], 'time_comp'] = time_comp

            else:
                pass

        comp_dup = len(transcript_comp[~np.isnan(transcript_comp['time_comp'])\
                       & (transcript_comp.duplicated(subset=['person', 'clean_offer_id', 'time_comp']))])

        print('Completed data joined!')

        assert (~np.isnan(transcript_comp['time_rec'])).sum() == len(transcript_received), 'Duplicate received timestamps'
        assert (~np.isnan(transcript_comp['time_view'])).sum() == len(transcript_viewed), 'Duplicate viewed timestamps'
        assert comp_dup == 0, 'Duplicate completed timestamps'

        print('Number of completed offers on records: ', len(transcript_completed.drop_duplicates()))
        print('Number of viewed AND completed offers: (only these are joined into the df) ',
              (~np.isnan(transcript_comp['time_comp'])).sum())
        print(' ')


        # Step 4: Join in transaction data
        # Note that as one transaction can complete several offers, this could be a one to many join
        transcript_trans = transcript_comp.merge(transcript_transact, how = 'outer', left_on = ['time_comp', 'person'], \
                           right_on = ['time', 'person'])
        transcript_trans = transcript_trans.rename({'time':'time_trans'}, axis = 1)

        unmatch_trans = transcript_trans[np.isnan(transcript_trans['time_rec'])]\
                        .drop(['time_rec', 'time_view', 'time_comp'], axis = 1)

        transcript_trans = transcript_trans.drop(unmatch_trans.index, axis = 0)

        print('Transaction data joined!')
        assert (~np.isnan(transcript_trans['time_rec'])).sum() == len(transcript_received), 'Duplicate received timestamps'
        assert (~np.isnan(transcript_trans['time_view'])).sum() == len(transcript_viewed), 'Duplicate viewed timestamps'
        assert (~np.isnan(transcript_trans['time_trans'])).sum() == (~np.isnan(transcript_comp['time_comp'])).sum(), \
        'Duplicate transaction timestamps'
        print('Number of transactions that do not join:', len(unmatch_trans)) 
        print(' ')

        # Step 5: Join in informational offer completion time stamps
        transcript_port = transcript_trans.copy()
        unprompted_purchase_idx = []

        for index, row in unmatch_trans.iterrows():

            user = row['person']
            time_trans = row['time_trans']
            amount = row['amount']

            criteria = ((transcript_port['person'] == user) &
                       (transcript_port['time_rec'] <= time_trans) &
                       (transcript_port['time_view'] <= time_trans) &
                       (transcript_port['effect_timeout'] >= time_trans) &
                       (transcript_port['offer_type']=='informational') &
                       (np.isnan(transcript_port['time_comp'])))

            trans_idx = transcript_port.loc[criteria].index

            if len(trans_idx)>0:
                idx = trans_idx[0]
                transcript_port.loc[idx, ['time_comp', 'time_trans', 'amount']] = time_trans, time_trans, amount

            else:        
                # purchase is not prompted by any offers
                unprompted_purchase_idx.append(index)

        transcript_port['amount'] = transcript_port['amount'].astype(float)
        unmatch_trans_final = unmatch_trans.loc[unprompted_purchase_idx][['person', 'time_trans', 'amount']]

        print('Completed info offer timestamps joined!')
        assert (~np.isnan(transcript_port['time_rec'])).sum() == len(transcript_received), 'Duplicate received timestamps'
        assert (~np.isnan(transcript_port['time_view'])).sum() == len(transcript_viewed), 'Duplicate viewed timestamps'
        print('Number of completed offers: ',(~np.isnan(transcript_port['time_trans'])).sum())
        print('Number of transactions not prompted by offers: ', len(unmatch_trans_final))

        return transcript_port, unmatch_trans_final

    def add_prior_col(self, main_df, unmatch_transactions_df):
        '''
        Add features regarding number of prior purchases not prompted by any offers, number of prior offers received,
        viewed or completed
        '''

        new_prior_col = ['num_purch_prior', 'avg_purch_amt_prior', 'num_rec_prior', 'num_view_prior', 'num_comp_prior']

        for col in new_prior_col:
            main_df[col] = float('NaN')

        for index, row in main_df.iterrows():
            person_num = row['person']
            time_offer = row['time_rec']

            criteria = ((unmatch_transactions_df['person'] == person_num) &
                       (unmatch_transactions_df['time_trans'] <= time_offer))

            prev_purch_df = unmatch_transactions_df.loc[criteria]

            if len(prev_purch_df) > 0:
                main_df.loc[index, 'num_purch_prior'] = len(prev_purch_df)
                main_df.loc[index, 'avg_purch_amt_prior'] = prev_purch_df['amount'].mean()

            else:
                pass

            criteria_2 = (main_df['person'] == person_num)
            actions = ['rec', 'view', 'comp']

            for act in actions:
                main_df.loc[index, 'num_'+act+'_prior'] = main_df.loc[criteria_2 &\
                                                                (main_df['time_'+act] < time_offer)]\
                                                                ['time_'+act].value_counts().sum()

        return main_df
    
    def feature_create(self, df_input):
        '''
        Feature engineering of final dataframe, including one-hot encoding, interaction terms, and feature creation
        '''

        df = df_input.copy()

        # Flag successful completions
        df['success'] = df['time_comp'].apply(lambda x:1 if ~np.isnan(x) else 0)

        # Calculate percentage of prior viewed and completed offers
        df['percent_view_prior'] = df['num_view_prior'] / df['num_rec_prior']
        df['percent_comp_prior'] = df['num_comp_prior'] / df['num_rec_prior']

        # Calculate frequency of prior purchases
        df['freq_purch_prior'] = df['time_rec'] / df['num_purch_prior']

        prior_col = ['num_purch_prior', 'avg_purch_amt_prior', 'num_rec_prior', 'num_view_prior', 'num_comp_prior',
                    'percent_view_prior', 'percent_comp_prior', 'freq_purch_prior']
        df[prior_col] = df[prior_col].fillna(0)
        df['amount'] = df['amount'].fillna(0)

        return df
    
    def create_interaction(self, df_X, treatment_col):
        '''
        Create dummy variables for treatment type and interaction term between treatment type and other covariates
        '''
        df = df_X.copy()

        df['offer_id_short'] = df[treatment_col].apply(lambda x:x[:4])
        df = pd.get_dummies(df, columns = ['offer_id_short'], prefix = 'offer')

        demo_cols = df_X.columns.to_list()
        demo_cols.remove(treatment_col)
        offer_dummy_col = [col for col in df.columns if col[:6] == 'offer_']

        for col_offer in offer_dummy_col:
            for col_demo in demo_cols:
                df[col_demo + '_' + col_offer[-4:]] = df[col_offer].astype(float) * df[col_demo].astype(float)

        df = df.drop([treatment_col], axis = 1)

        return df, df.columns
    
    
class EDAplot:
    
    def __init__(self):
        pass
    
    def treatment_dist_plot(self, df, var, lim = None):
        '''
        Plotting overlapping distribution plots of covariates to investigate their distribution across treatment groups
        '''

        offer_id_list = df['clean_offer_id'].unique()

        if lim == None:
            for offid in offer_id_list:
                sns.kdeplot(df.loc[df['clean_offer_id']==offid][var], label = offid[:4])
        else:
            for offid in offer_id_list:
                sns.kdeplot(df.loc[df['clean_offer_id']==offid][var], label = offid[:4], clip = lim)
                

class TreatmentModelling:
    
    def __init__(self, df):
        offer_ids = df['clean_offer_id'].unique()
        self.class_offer_list = offer_ids
    
    def calc_lift(self, df):
        '''
        Calculate lift compared to historical uptake, purchase frequency and amount
        '''

        offer_ids = self.class_offer_list
        lift_df = {'offer': [],
                  'perc_exposed': [],
                  'perc_exposed_completed':[],
                  'perc_exposed_prior_purch': [],
                  'perc_exposed_freq':[],
                  'perc_exposed_prior_freq': [],
                  'perc_exposed_completed_netamt': [],
                  'perc_exposed_prior_amt': []
                  }

        for offer in offer_ids:
            offer_df = df.loc[(df['clean_offer_id'] == offer)]
            exposed_df = offer_df.loc[~np.isnan(df['time_view'])]
            exposed_df.loc[(exposed_df['success']==1),'comp_netamt'] = exposed_df['amount'] - exposed_df['reward']
            exposed_df.loc[(exposed_df['num_purch_prior']!=0),'prior_freq'] = \
                exposed_df['time_rec'] / exposed_df['num_purch_prior']
            exposed_df.loc[(exposed_df['num_purch_prior']!=0),'new_freq'] = \
                exposed_df['time_comp'] / (exposed_df['num_purch_prior'] + exposed_df['success'])

            lift_df['offer'].append(offer)
            lift_df['perc_exposed'].append(len(exposed_df)/ len(offer_df))
            lift_df['perc_exposed_completed'].append(exposed_df['success'].sum()/ len(exposed_df))
            lift_df['perc_exposed_prior_purch'].append((exposed_df['num_purch_prior']!=0).sum()/ len(exposed_df))
            lift_df['perc_exposed_completed_netamt'].append(exposed_df['comp_netamt'].mean())
            lift_df['perc_exposed_prior_amt'].append(exposed_df.loc[(exposed_df['comp_netamt']!=0), 
                                                                    'avg_purch_amt_prior'].mean())
            lift_df['perc_exposed_prior_freq'].append(exposed_df['prior_freq'].mean())
            lift_df['perc_exposed_freq'].append(exposed_df['new_freq'].mean())

        output_df = pd.DataFrame(lift_df)
        output_df['lift_comp'] = output_df['perc_exposed_completed'] - output_df['perc_exposed_prior_purch']
        output_df['incre_amt'] = (output_df['perc_exposed_completed_netamt'] - output_df['perc_exposed_prior_amt'])/ \
                                 output_df['perc_exposed_prior_amt']
        output_df['incre_freq'] = (output_df['perc_exposed_freq'] - output_df['perc_exposed_prior_freq'])/ \
                                 output_df['perc_exposed_prior_freq']
        output_df['expected_uplift'] = output_df['perc_exposed'] * output_df['perc_exposed_completed'] * \
                                     (output_df['perc_exposed_completed_netamt'] - output_df['perc_exposed_prior_amt'])
        output_df['prob_comp'] = output_df['perc_exposed'] * output_df['perc_exposed_completed']
        output_df = output_df[['offer', 'prob_comp', 'expected_uplift']].sort_values(by = 'expected_uplift', ascending = False)
        
        return output_df
    
    def choose_model(self, X, y, model_list, model_type):
        '''
        Running a list of models (in default parameters) to choose the most suitable one. It outputs the scoring
        of either regression or classification
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, shuffle = True)

        if model_type == 'class':
            score_result = {'Model':[],
                            'Model_short':[],
                            'Test Precision': [],
                            'Train Precision': [],
                            'Test Recall': [],
                            'Train recall': [],
                            'Test ROC AUC': [],
                            'Train ROC AUC': []}
        elif model_type == 'reg':
            score_result = {'Model':[],
                            'Model_short':[],
                            'Test MSE': [],
                            'Train MSE': [],
                            'Test R2': [],
                            'Train R2':[]}

        for model in model_list:
            score_result['Model'].append(model)
            score_result['Model_short'].append(str(model)[:15])
            train_model = model
            train_model.fit(X_train, y_train)

            y_train_pred = train_model.predict(X_train)
            y_test_pred = train_model.predict(X_test)

            if model_type == 'class':
                score_result['Test Precision'].append(precision_score(y_test, y_test_pred))
                score_result['Train Precision'].append(precision_score(y_train, y_train_pred))
                score_result['Test Recall'].append(recall_score(y_test, y_test_pred))
                score_result['Train recall'].append(recall_score(y_train, y_train_pred))
                score_result['Test ROC AUC'].append(roc_auc_score(y_test, y_test_pred))
                score_result['Train ROC AUC'].append(roc_auc_score(y_train, y_train_pred))
            elif model_type == 'reg':
                score_result['Test MSE'].append(mean_squared_error(y_test, y_test_pred))
                score_result['Train MSE'].append(mean_squared_error(y_train, y_train_pred))
                score_result['Test R2'].append(r2_score(y_test, y_test_pred))
                score_result['Train R2'].append(r2_score(y_train, y_train_pred))

        for k, v in score_result.items():
            if k == 'Model':
                pass
            else:
                print(k, v)

        print('')

        return score_result
    
    def gridsearch_pipelines(self, X, y, m1, p1, scoring_list, refit_score, m2 = None, p2 = None, m3 = None, \
                             p3 = None, m4 = None, p4 = None, m5 = None, p5 = None):
        '''
        Fit training data to up to five different models and optimise each across various hyperparameters
        using grid search CV. This function outputs the mean score of each model and the best parameter of each model
        '''
        
        saved_results = {'model':[],
                        'best_param':[],
                        'best_score':[],
                        'best_estimator':[],
                        'cv_results': []}
        model_list = [m1, m2, m3, m4, m5]
        param_list = [p1, p2, p3, p4, p5]

        for model, param in zip(model_list, param_list):
            if model != None:
                gs_cv = GridSearchCV(model, param, scoring = scoring_list, refit = refit_score,
                                     cv = 3, verbose = 10, n_jobs = -1, return_train_score = True)
                gs_cv.fit(X, y)

                saved_results['model'].append(str(model)[:10])
                saved_results['best_param'].append(gs_cv.best_params_)
                saved_results['best_score'].append(gs_cv.best_score_)
                saved_results['best_estimator'].append(gs_cv.best_estimator_)
                saved_results['cv_results'].append(gs_cv.cv_results_)

                score_keys = ['mean_test_' + i for i in scoring_list] + ['mean_train_' + i for i in scoring_list]

                print('Best Estimator of ', str(model)[:10], ' :', gs_cv.best_params_)
                print('Best refit test score', gs_cv.best_score_)
                print('Mean test scores')
                for score_method in score_keys:
                    print(score_method, np.array(gs_cv.cv_results_[score_method]).mean())
                print(' ')

            else:
                pass

        return saved_results
    
    
    def train_model(self, X, y, model, model_type):
        '''
        Split data into train and test and train one model on the training set. 
        
        INPUT:
        model_type - can be either 'class' for classification model or 'reg' for regression model
        '''

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train.values, y_train.values)
        y_train_pred = model.predict(X_train.values)
        y_test_pred = model.predict(X_test.values)

        if model_type == 'class':
            score_result = {'Test Precision': precision_score(y_test, y_test_pred),
                            'Train Precision': precision_score(y_train, y_train_pred),
                            'Test Recall': recall_score(y_test, y_test_pred),
                            'Train recall': recall_score(y_train, y_train_pred),
                            'Test ROC AUC': roc_auc_score(y_test, y_test_pred),
                            'Train ROC AUC': roc_auc_score(y_train, y_train_pred)}
        elif model_type == 'reg':
            score_result = {'Test MSE': mean_squared_error(y_test, y_test_pred),
                            'Train MSE': mean_squared_error(y_train, y_train_pred),
                            'Test R2': r2_score(y_test, y_test_pred),
                            'Train R2': r2_score(y_train, y_train_pred)}

        for k, v in score_result.items():
            print(k, v)

        return model
    
    def export_model(self, model, model_name):
        '''
        Exports model to pickle file in the same folder with specified model_name 
        (which should end in .pkl)
        '''

        joblib.dump(value=model, filename = model_name)
        
    def predict_prob(self, comp_model, X, X_interact_col):
        '''
        Load model to predict probability of completing the offer given demographic and behaviour stats
        '''
        # Prepare X df
        X_new = pd.concat([X]*10, ignore_index=True) 
        X_new['clean_offer_id'] = list(self.class_offer_list) #change to self

        dpreprocess = RawPreprocess()
        X_new_interact, X_new_interact_cols = dpreprocess.create_interaction(X_new, 'clean_offer_id')
        X_new_interact = X_new_interact[X_interact_col]

        # Import classifier model
        classifier = joblib.load(comp_model)
        pred_result = np.array(classifier.predict_proba(X_new_interact.as_matrix()))[:,0]
        X_new_interact['pred_result'] = pred_result.ravel()
        X_new_interact['clean_offer_id'] = X_new['clean_offer_id']

        X_return = X_new_interact[['clean_offer_id', 'pred_result']]

        return X_return
    
    def predict_spend(self, uplift_model_list, nonoff_spend_model, X):
        '''
        Load model to predict net revenue uplift of completing the offer given demographic and behaviour stats
        '''
        X_nonoff = X[['age', 'income', 'null_info', 'days_joined', 'gender_F', 'gender_M', 'gender_O']]
        # Predict spend without offer
        nonoff_reg = joblib.load(nonoff_spend_model)
        nonoff_result = nonoff_reg.predict(X_nonoff.as_matrix())[0]

        # Predict spend with offer
        clean_offer_id = list(self.class_offer_list)
        uplift_result = []
        for off_id in self.class_offer_list:
            reg_model_name = [mod for mod in uplift_model_list if off_id[:4] in mod][0]
            reg_model = joblib.load(reg_model_name, 'rb')
            uplift_result.append(reg_model.predict(X.as_matrix())[0])

        X_return = pd.DataFrame({
            'clean_offer_id': clean_offer_id,
            'uplift_rev': uplift_result
        })

        X_return['uplift_rev'] = X_return['uplift_rev'] - nonoff_result

        return X_return 
    
    def recommend_offer(self, offer, profile, cleandata, unmatch_trans, comp_model, 
                        uplift_model_list, nonoff_spend_model, entry_type, X_interact_col, 
                        customer_id = None, customer_detail = None):
        '''
        Enter customer details and return at most top three most relevant offers

        INPUTS:
        profile_name - csv of cleaned profile data
        entry_type - can be 'customer_id' for existing customer or 'new_customer' for new customer who is
        not in existing
        customer_detail - dict in the format of {'age': value, 'income': value, 'gender': 'F', 
        'join_date': YYYYmmdd}
        '''
        offer_df = pd.read_csv(offer)
        profile_df = pd.read_csv(profile)
        data_df = pd.read_csv(cleandata)
        unmatch_trans_df = pd.read_csv(unmatch_trans)

        if entry_type == 'customer_id':
            try:
                X_col = ['age', 'income', 'null_info', 'days_joined', 'gender_F', 'gender_M', 
                         'gender_O']
                X = profile_df.loc[profile_df['id']==customer_id][X_col]
                for col in X.columns:
                    print(col, X[col].values[0])
                print(' ')

                # Recalculate days since joined
                max_prof_date = pd.to_datetime(20180726, format='%Y%m%d').date()
                X['days_joined'] = X['days_joined'] + (date.today() - max_prof_date).days

                # Calculate prior purchase information
                customer_unmatch = unmatch_trans_df.loc[unmatch_trans_df['person']==customer_id]

                if len(customer_unmatch)>0:
                    X['freq_purch_prior'] = 714/ len(customer_unmatch)
                    X['avg_purch_amt_prior'] = customer_unmatch['amount'].mean()
                else:
                    X['freq_purch_prior'], X['avg_purch_amt_prior'] = 0, 0

                # Calculate prior offer completion information
                prior_df = data_df.loc[data_df['person']==customer_id]
                if len(prior_df)>0:
                    X['num_rec_prior'] = len(prior_df)
                    X['percent_view_prior'] = (~np.isnan(prior_df['time_view'])).sum() / len(prior_df)
                    X['percent_comp_prior'] = (~np.isnan(prior_df['time_comp'])).sum() / len(prior_df)
                else:
                    X['num_rec_prior'], X['percent_view_prior'], X['percent_comp_prior'] = 0, 0, 0   
                
            except:
                print('Please enter correct customer_id and start again')
                return None

        elif entry_type == 'new_customer':
            if customer_detail != None:
                try:
                    # Input new customer details
                    X = pd.DataFrame([[customer_detail['age'], customer_detail['income']]], columns = ['age', 'income'])

                    join_date = pd.to_datetime(customer_detail['join_date'], format='%Y%m%d').date()
                    X['days_joined'] = (date.today() - join_date).days

                    X['gender_' + str(customer_detail['gender'])] = 1
                    for i in ['M', 'F', 'O']:
                        if i == customer_detail['gender']:
                            pass
                        else:
                            X['gender_' + i] = 0

                    X['null_info'], X['freq_purch_prior'], X['avg_purch_amt_prior'], X['num_rec_prior'], \
                    X['percent_view_prior'],X['percent_comp_prior'] = 0, 0, 0, 0, 0, 0

                except:
                    print('You might have missed some customer information fields. Please try again')
                    return None

            else:
                # If no customer detail is given, return default offer with highest expected 
                # uplift in net profit

                proposed_offer = ['fafdcd668e3743c1bb461111dcafc2a4', 
                                  '2298d6c36e964ae4a3e7e9706d1fb8c2',
                                  'f19421c1d4aa40978ebb69ca19b0e20d']
                probabilty_completion = [0.61, 0.57, 0.48]
                netrev_uplift = [7.51, 5.22, 6.10]
                expect_netrev_uplift = [4.58, 2.98, 2.93]

                proposed_df = pd.DataFrame({'Proposed Offer': proposed_offer,
                                            'Probability Of Completion': probabilty_completion,
                                            'NetRev Uplift': netrev_uplift,
                                            'Expected NetRev Uplift': expect_netrev_uplift})

                return proposed_df

        X = X[['age', 'income', 'null_info', 'days_joined', 'gender_F', 'gender_M', 
               'gender_O', 'freq_purch_prior', 'avg_purch_amt_prior', 'num_rec_prior', 
               'percent_view_prior', 'percent_comp_prior']]

        pred_prob_df = self.predict_prob(comp_model, X, X_interact_col)
        pred_spend_df = self.predict_spend(uplift_model_list, nonoff_spend_model, X)

        proposed_df = pred_prob_df.merge(pred_spend_df, on = 'clean_offer_id') \
                      .merge(offer_df[['id','difficulty']], left_on = 'clean_offer_id', right_on = 'id')

        proposed_df['NetRev Uplift'] = proposed_df['uplift_rev'] - proposed_df['difficulty']
        proposed_df['Expected NetRev Uplift'] = proposed_df['NetRev Uplift'] * proposed_df['pred_result']
        proposed_df = proposed_df.loc[proposed_df['Expected NetRev Uplift']>=0]\
                      .sort_values(by = 'Expected NetRev Uplift', ascending = False)\
                      .rename(columns={"clean_offer_id": "Proposed Offer", "pred_result": "Probability Of Completion"})\

        return proposed_df[['Proposed Offer', 'Probability Of Completion', 'NetRev Uplift', 'Expected NetRev Uplift']].head(3)

