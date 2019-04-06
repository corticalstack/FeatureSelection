import pandas as pd
from sklearn.datasets import load_wine
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


class FeatureSelection:
    def __init__(self):
        self.wine = None
        self.df_wine = None
        self.X = None
        self.y = None
        self.random_state = 20
        self.target = None
        self.load_data()
        self.set_data()
        self.univariate_feature_selection()
        self.recursive_feature_elimination()
        self.pca()
        self.extra_trees_classifier()
        self.random_forest_classifier()

    def load_data(self):
        self.wine = load_wine()

    def set_data(self):
        self.df_wine = pd.DataFrame(self.wine.data, columns=self.wine.feature_names)
        self.X = self.df_wine.values
        self.y = pd.Series(self.wine.target)

    # Univariate statistical Chi-squared
    def univariate_feature_selection(self):
        print('\n', '_' * 40, 'Univariate feature selection with chi-squared', '_' * 40)
        kbest = SelectKBest(score_func=chi2, k=4)
        fit = kbest.fit(self.X, self.y)
        print(fit.scores_)
        cols = kbest.get_support()
        features_selected = self.df_wine.columns[cols]
        print(features_selected)

    # Recursive Feature Elimination
    def recursive_feature_elimination(self):
        print('\n\n', '_' * 40, 'Recursive feature selection with logistic regression', '_' * 40)
        model = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=600)
        rfe = RFE(model, 4)
        fit = rfe.fit(self.X, self.y)
        print("Num Features: {}".format(fit.n_features_))
        print("Selected Features: {}".format(fit.support_))
        print("Feature Ranking: {}".format(fit.ranking_))
        features_selected = self.df_wine.columns[fit.support_]
        print(features_selected)

    # PCA (Principal Component Analysis)
    def pca(self):
        print('\n\n', '_' * 40, 'Principal Component Analysis', '_' * 40)
        pca = PCA(n_components=4)
        fit = pca.fit(self.X)
        # summarize components
        print("Explained Variance: {}".format(fit.explained_variance_ratio_))
        print(fit.components_)

    # Extra Trees Classifier
    def extra_trees_classifier(self):
        print('\n\n', '_' * 40, 'Extra Trees Classifier', '_' * 40)
        model = ExtraTreesClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(self.X, self.y)
        print(list(zip(self.df_wine.columns, model.feature_importances_)))

    # Random Forest Classifier
    def random_forest_classifier(self):
        print('\n\n', '_' * 40, 'Random Forest Classifier', '_' * 40)
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(self.X, self.y)
        print(list(zip(self.df_wine.columns, model.feature_importances_)))


feature_selection = FeatureSelection()
