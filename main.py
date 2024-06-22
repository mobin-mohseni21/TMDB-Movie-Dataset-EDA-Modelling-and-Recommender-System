import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash
from flask import Flask
from pre_processor import PreProcessor, FeatureSelection
from dash_bootstrap_templates import load_figure_template
from dashboard import DashboardLayout, CallbackManager
from models import MachineLearningClassifier


movies = pd.read_csv('tmdb_reformatted_movies.csv')
credit = pd.read_csv('tmdb_reformatted_credits.csv')
data_frame = pd.merge(left=movies, right=credit, how='inner', on='id')
pre_processor = PreProcessor(data_frame=data_frame)
pre_processor.set_index('id')
pre_processor.drop_na(axis=1, how='any')
pre_processor.data_frame.rename(columns={'company_name': 'Producer_Company', 'genre': 'Genre', 'cast': 'Cast'}
                                , inplace=True)
pre_processor.data_frame['release_date'] = pd.to_datetime(pre_processor.data_frame['release_date'],
                                                          format='%Y-%m-%d')
data_frame = pre_processor.data_frame.copy()
pre_processor.data_frame['profit'] = pre_processor.data_frame['revenue'] - pre_processor.data_frame['budget']
pre_processor.data_frame['profit'] = pre_processor.data_frame['profit'].apply(lambda x: 1 if x > 0 else 0)
target = pre_processor.data_frame.pop('profit')

print(data_frame['iso'].unique())
pre_processor.drop_columns(columns=['Country', 'release_date', 'title', 'department', 'job', 'name'])
pre_processor.normalize()
pre_processor.one_hot_encoder(columns=['Genre'])

feature_selector = FeatureSelection(pre_processor.data_frame)
features = feature_selector.data_frame.select_dtypes(include=['number']).columns
# feature_selector.extend_data_by_hdbscan(features=features)
# print(feature_selector.data_frame)
feature_selector.add_log_transform(columns=['budget'])
feature_selector.reduction_dimension_by_pca(scaled_columns=features)
# featuer_selector.calculate_mutual_inf_class(target=target, number_of_features=12)



#%%
data = list()
model = MachineLearningClassifier(data_frame=feature_selector.data_frame, target=target)
model.train_test_split()
print('train_test split done')
model.fit_xgboost_classifier()
print('train_xgboost_classifier done')
data.append(model.metrics())
model.fit_logistic_regression_classifier()
print('train_logistic_regression_classifier done')
data.append(model.metrics())
data.append(model.metrics())
model.fit_decision_tree_classifier()
print('train_decision_tree_classifier done')
data.append(model.metrics())
model.fit_random_forest_classifier()
print('train_random_forest_classifier done')
data.append(model.metrics())


df = pd.DataFrame(data=data)
print(df)
load_figure_template('SLATE')
# print(pre_processor.data_frame)
movies = pd.read_csv('tmdb_5000_movies.csv')
if __name__ == "__main__":
    server = Flask(__name__)
    server.config.update(
        broker_url='amqp://guest:guest@localhost//',
        result_backend='rpc://')
    app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE], server=server)
    dashboard_layout = DashboardLayout(app, data_frame)
    callback_manager = CallbackManager(app, data_frame, machine_learning_data_frame=df, target=target,  recommender_data= movies)
    app.run_server(debug=True,  port=8080)
