import warnings
import pandas as pd
import seaborn as sb
import shap
import pickle
from palmerpenguins import load_penguins
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split


warnings.simplefilter(action='ignore', category=FutureWarning)
d_peng = load_penguins()
d_peng.dropna(inplace=True)
feature_list = [
    'island', 'bill_length_mm', 'bill_depth_mm',
    'flipper_length_mm', 'body_mass_g', 'sex']
features = d_peng[feature_list]
features = pd.get_dummies(features)
out, uniques = pd.factorize(d_peng.species)

X_train, X_val, y_train, y_val = train_test_split(
    features, out, test_size=0.8)
rfc = RFC()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_val)
score = accuracy_score(y_pred, y_val)
print(f'Model accuracy is {score:.2f}')
explainer = shap.TreeExplainer(rfc)

with open('random_forest_penguins.pkl', 'wb') as rfc_pkl:
    pickle.dump(rfc, rfc_pkl)
with open('label_mapper.pkl', 'wb') as map_pkl:
    pickle.dump(uniques, map_pkl)
with open('rf_explainer.pkl', 'wb') as xai_pkl:
    pickle.dump(explainer, xai_pkl)

