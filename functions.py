#import all necessary packages for the code below
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.reductions import EqualizedOdds
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import DemographicParity, EqualizedOdds
from fairlearn.reductions import ExponentiatedGradient

def generate_boxplots(metrics_mitigated: List[float], 
                      n_size: List[int],
                      metric: str, 
                      method: str, 
                      file_path: str,
                      dataset: str,
                      clf: str
    ):
    
    fig, ax = plt.subplots(figsize=(5, 6))
    
    # Customize the boxplot
    boxprops = dict(linewidth=2, color='darkblue')  # Box properties
    whiskerprops = dict(linewidth=1.5, linestyle='--', color='gray')  # Whisker properties
    medianprops = dict(linewidth=2.5, color='black')  # Median properties
    ax.boxplot(metrics_mitigated, boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops)
    
    # Customize the x-axis tick labels
    ax.set_xticklabels(['n = {}'.format(n_size[0]), 'n = {}'.format(n_size[1]), 'n = {}'.format(n_size[2])], fontsize=12)

    # Set title, labels, and limitsW
    ax.set_title('{}'.format(metric), fontsize=20)
    ax.set_xlabel('Sample size (n)', fontsize=16)
    
    if metric == 'Demographic parity':
        ax.set_ylabel('DP gap', fontsize=14)
    elif metric == 'Equalized odds':
        ax.set_ylabel('EO gap', fontsize=14)
    elif metric == 'AUC':
        ax.set_ylabel('AUC', fontsize=14)
    else:
        print("Choose a valid metric")
        return
    
    min_value = np.min([np.min(m) for m in metrics_mitigated])
    max_value = np.max([np.max(m) for m in metrics_mitigated])
    ax.set_ylim(-0.05 + min_value, max_value + 0.05)

    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust the layout to prevent labels from getting cut off
    fig.tight_layout()

    # Save the figure
    fig.savefig(file_path, dpi = 200)

    # Display the plot
    plt.show()
    
def print_confidence_intervals(df1: pd.DataFrame, 
                               df2: pd.DataFrame, 
                               df3: pd.DataFrame
) -> None:
    print('Confidence intervals for size 1')
    print('AUC: [{:.2f},{:.2f}]'.format(df1.auc.quantile(0.025), df1.auc.quantile(0.975)))
    print('DP: [{:.2f},{:.2f}]'.format(df1.dp.quantile(0.025), df1.dp.quantile(0.975)))
    print('EO: [{:.2f},{:.2f}]'.format(df1.eo.quantile(0.025), df1.eo.quantile(0.975)))
    
    print()
    print('Confidence intervals for size 2')
    print('AUC: [{:.2f},{:.2f}]'.format(df2.auc.quantile(0.025), df2.auc.quantile(0.975)))
    print('DP: [{:.2f},{:.2f}]'.format(df2.dp.quantile(0.025), df2.dp.quantile(0.975)))
    print('EO: [{:.2f},{:.2f}]'.format(df2.eo.quantile(0.025), df2.eo.quantile(0.975)))
    
    print()
    print('Confidence intervals for size 3')
    print('AUC: [{:.2f},{:.2f}]'.format(df3.auc.quantile(0.025), df3.auc.quantile(0.975)))
    print('DP: [{:.2f},{:.2f}]'.format(df3.dp.quantile(0.025), df3.dp.quantile(0.975)))
    print('EO: [{:.2f},{:.2f}]'.format(df3.eo.quantile(0.025), df3.eo.quantile(0.975)))

def correlation_heatmap(X_train, sensitive_feature: str, mode: int, file_path: str):
    cr = CorrelationRemover(sensitive_feature_ids=[sensitive_feature])
    X_train_cr = cr.fit_transform(X_train)
    df_train_cr = pd.DataFrame(X_train_cr, columns=X_train.drop(columns=[sensitive_feature]).columns)
    df_train_cr.insert(loc=4, column=sensitive_feature, value=X_train[sensitive_feature])
    
    plt.figure(figsize=(8, 8))
    cmap = sns.color_palette("PuBu", as_cmap=True)
    
    df_original = X_train.iloc[:,0:6]
    df_removed = df_train_cr.iloc[:,0:6]
    
    if mode == 0:
        title = 'Original data'
        ax = sns.heatmap(df_original.corr(), annot=True, fmt=".2f", cmap=cmap)
        
    elif mode == 1:
        title = 'After correlation remover'
        ax = sns.heatmap(df_removed.corr(), annot=True, fmt=".2f", cmap=cmap)
        
    else:
        "choose a valid mode"
        return
    
    ax.set_title(title, fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90) 
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(file_path, dpi = 200, bbox_inches='tight')
    plt.show()  
    
def bootstrap_correlation_remover(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    A_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    A_test: np.ndarray,
    sensitive_feature: str, 
    B: int, 
    n_size: List[int],
    clf: str,
    base_path: str
) -> None:
    
    assert(sensitive_feature in X_train.columns)
    
    n_train = len(X_train)
    ind_train = np.arange(n_train)
    
    for n in tqdm(n_size):
        auc_boot_mitigated = []; eo_boot_mitigated = []; dp_boot_mitigated = []
        df_mitigated = pd.DataFrame()
        
        sample_ind = np.random.choice(ind_train, size = n, replace=False)
        X_train_sample = X_train.iloc[sample_ind,:]
        y_train_sample = y_train.iloc[sample_ind]
        
        n_train_sample = len(X_train_sample)
        ind_train_sample = np.arange(n_train_sample)

        for b in tqdm(range(B)):
            try:
                boot_ind = np.random.choice(ind_train_sample, size = n, replace=True)
                X_train_boot = X_train_sample.iloc[boot_ind,:]
                y_train_boot = y_train_sample.iloc[boot_ind]
                
                cr = CorrelationRemover(sensitive_feature_ids=[sensitive_feature])
                X_train_cr = cr.fit_transform(X_train_boot)
                
                if clf == 'lr':
                    classifier = LogisticRegression(random_state=42, max_iter = 1000)
                    
                elif clf == 'rf':
                    classifier = RandomForestClassifier(random_state=42)
                else:
                    print("choose a valid classifier")
                    return
                
                classifier.fit(X_train_cr, y_train_boot)
                X_test_cr = cr.transform(X_test)

                y_prob = classifier.predict_proba(X_test_cr)[:,1]
                y_pred = classifier.predict(X_test_cr)

                auc_mitigated = roc_auc_score(y_test, y_prob)
                eo_mitigated = equalized_odds_difference(y_test, y_pred, sensitive_features=A_test)
                dp_mitigated = demographic_parity_difference(y_test, y_pred, sensitive_features=A_test)

                auc_boot_mitigated.append(auc_mitigated)
                eo_boot_mitigated.append(eo_mitigated )
                dp_boot_mitigated.append(dp_mitigated)

            except:
                print("not enough labels")

        df_mitigated['auc'] = auc_boot_mitigated
        df_mitigated['eo'] = eo_boot_mitigated
        df_mitigated['dp'] = dp_boot_mitigated

        path = base_path+'pre_metrics_nsize_{}_{}'.format(n, clf)
        df_mitigated.to_csv(path, index = False)

def bootstrap_exp_gradient(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    A_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    A_test: np.ndarray,
    fair_constraint: str, 
    B: int, 
    n_size: List[int],
    clf: str,
    base_path: str,
) -> None:
    
    n_train = len(X_train)
    ind_train = np.arange(n_train)
    print("new version")
    
    for n in tqdm(n_size):
        auc_boot_mitigated = []; eo_boot_mitigated = []; dp_boot_mitigated = []
        df_mitigated = pd.DataFrame()
        
        sample_ind = np.random.choice(ind_train, size = n, replace=False)
        X_train_sample = X_train.iloc[sample_ind,:]
        y_train_sample = y_train.iloc[sample_ind]
        A_train_sample = A_train.iloc[sample_ind]
        
        n_train_sample = len(X_train_sample)
        ind_train_sample = np.arange(n_train_sample)

        for b in tqdm(range(B)):
            try:
                boot_ind = np.random.choice(ind_train_sample, size = n, replace=True)
                X_train_boot = X_train_sample.iloc[boot_ind,:]
                y_train_boot = y_train_sample.iloc[boot_ind]
                A_train_boot = A_train_sample.iloc[boot_ind]
                
                if fair_constraint == "equalized_odds":
                    constraint = EqualizedOdds()
                elif fair_constraint == "demographic_parity":
                    constraint = DemographicParity()
                else:
                    print("fair_constraint must be either equalized_odds or demographic_parity")
                    return
                
                if clf == 'lr':
                    classifier = LogisticRegression(random_state=42, max_iter = 1000)
                elif clf == 'rf':
                    classifier = RandomForestClassifier(random_state=42)
                else:
                    print("choose a valid classifier")
                    return
                
                mitigator = ExponentiatedGradient(classifier, constraint)
                mitigator.fit(X = X_train_boot, y = y_train_boot, sensitive_features=A_train_boot)

                y_prob = mitigator._pmf_predict(X_test)[:,1]
                y_pred = mitigator.predict(X_test)

                auc_mitigated = roc_auc_score(y_test, y_prob)
                eo_mitigated = equalized_odds_difference(y_test, y_pred, sensitive_features=A_test)
                dp_mitigated = demographic_parity_difference(y_test, y_pred, sensitive_features=A_test)

                auc_boot_mitigated.append(auc_mitigated)
                eo_boot_mitigated.append(eo_mitigated )
                dp_boot_mitigated.append(dp_mitigated)

            except:
                print("not enough labels")

        df_mitigated['auc'] = auc_boot_mitigated
        df_mitigated['eo'] = eo_boot_mitigated
        df_mitigated['dp'] = dp_boot_mitigated

        path = base_path+'in_metrics_nsize_{}_{}_{}'.format(n, fair_constraint, clf)
        df_mitigated.to_csv(path, index = False)
        
def bootstrap_threshold_optimizer(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    A_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    A_test: np.ndarray,
    fair_constraint: str, 
    B: int, 
    n_size: List[int],
    clf: str,
    base_path: str,
) -> None:
    
    assert(fair_constraint in ["equalized_odds", "demographic_parity"])
    print("new version 4")
    n_train = len(X_train)
    ind_train = np.arange(n_train)
    
    for n in tqdm(n_size):
        auc_boot_mitigated = []; eo_boot_mitigated = []; dp_boot_mitigated = []
        df_mitigated = pd.DataFrame()
    
        sample_ind = np.random.choice(ind_train, size=n, replace=False)
        X_train_sample = X_train.iloc[sample_ind,:]
        y_train_sample = y_train.iloc[sample_ind]
        A_train_sample = A_train.iloc[sample_ind]
        
        n_train_sample = len(X_train_sample)
        ind_train_sample = np.arange(n_train_sample)

        for b in tqdm(range(B)):
            try:
                boot_ind = np.random.choice(ind_train_sample, size = n, replace=True)
                X_train_boot = X_train_sample.iloc[boot_ind,:]
                y_train_boot = y_train_sample.iloc[boot_ind]
                A_train_boot = A_train_sample.iloc[boot_ind]
                
                if clf == 'lr':
                    classifier = LogisticRegression(random_state=42, max_iter = 1000)
                elif clf == 'rf':
                    classifier = RandomForestClassifier(random_state=42)
                elif clf == 'adaboost':
                    classifier = AdaBoostClassifier(random_state=42)
                else:
                    print("choose a valid classifier")
                    return

                classifier.fit(X_train_boot, y_train_boot)

                post_est = ThresholdOptimizer(
                    estimator=classifier,
                    constraints=fair_constraint,
                    objective="balanced_accuracy_score",
                    prefit=True,
                    predict_method="predict_proba",
                )

                post_est.fit(X=X_train_boot, y=y_train_boot, sensitive_features=A_train_boot)

                post_pred = post_est.predict(X_test, sensitive_features=A_test)
                post_pred_proba = post_est._pmf_predict(X_test, sensitive_features=A_test)[:,1]

                auc_mitigated = roc_auc_score(y_test, post_pred_proba)
                eo_mitigated = equalized_odds_difference(y_test, post_pred, sensitive_features=A_test)
                dp_mitigated = demographic_parity_difference(y_test, post_pred, sensitive_features=A_test)

                auc_boot_mitigated.append(auc_mitigated)
                eo_boot_mitigated.append(eo_mitigated )
                dp_boot_mitigated.append(dp_mitigated)

            except:
                print("not enough labels")

        df_mitigated['auc'] = auc_boot_mitigated
        df_mitigated['eo'] = eo_boot_mitigated
        df_mitigated['dp'] = dp_boot_mitigated

        path = base_path+'post_metrics_nsize_{}_{}_{}'.format(n, fair_constraint, clf)
        df_mitigated.to_csv(path, index = False)
