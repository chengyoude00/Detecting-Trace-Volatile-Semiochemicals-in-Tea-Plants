import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, \
    StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_excel("gas_test_2.xls")

# Get the list of gas types
gas_types = data['label'].unique()


# Function to define models and parameter grids
def get_models_and_params():
    models_regression = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "MLP": MLPRegressor(max_iter=3000, learning_rate='adaptive', hidden_layer_sizes=(100,),
                            learning_rate_init=0.001),
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "ExtraTrees": ExtraTreesRegressor(random_state=42),
        # add StackingRegressor models
        "StackingRegressor": StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(random_state=42)),
                ('xgb', XGBRegressor(random_state=42)),
                ('mlp', MLPRegressor(max_iter=3000, learning_rate='adaptive', hidden_layer_sizes=(100,),
                                     learning_rate_init=0.001)),
                ('svr', SVR()),
                ('knn', KNeighborsRegressor()),
                ('dt', DecisionTreeRegressor()),
                ('ada', AdaBoostRegressor(random_state=42)),
                ('gb', GradientBoostingRegressor(random_state=42)),
                ('et', ExtraTreesRegressor(random_state=42))
            ],
            final_estimator=LinearRegression(),
            cv=5
        )
    }

    regression_param_grid = {
        "RandomForest": {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]},
        "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6, 10]},
        "MLP": {'hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)], 'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01], 'alpha': [0.0001, 0.001, 0.01]},
        "SVR": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        "KNN": {
            'n_neighbors': [k for k in [3, 5, 7] if k <= len(data)],  # 限制 n_neighbors 不超过样本数
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        },
        "DecisionTree": {'max_depth': [5, 10, 20, None]},
        "AdaBoost": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
        "GradientBoosting": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]},
        "ExtraTrees": {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    }
    return models_regression, regression_param_grid


models = {}
predictions = {}


# Iterate through each gas type, train the model separately, and make predictions
for gas_type in gas_types:
    print(f"Training on gas types {gas_type} ...")
    subset = data[data['label'] == gas_type]

    # Features and Target Variable    
    X = subset.drop(columns=['label', 'Con'])
    y = subset['Con']

    # Dataset Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get models and parameter grids
    models_regression, param_grid = get_models_and_params()
    best_model, best_score, best_model_name = None, float('inf'), ""

    # Dynamically select cross-validation strategy, ensuring n_splits does not exceed the number of samples
    n_splits = min(5, len(X_train))
    cv = KFold(n_splits=n_splits) if n_splits > 1 else LeaveOneOut()

    # Iterate through each model, perform hyperparameter tuning, and select the best model
    for model_name, model in models_regression.items():
        grid_search = GridSearchCV(model, param_grid.get(model_name, {}), cv=cv, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Use the best parameters model to predict and evaluate
        y_val_pred = grid_search.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2 = r2_score(y_val, y_val_pred) if len(y_val) > 1 else None

        print(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2 if r2 is not None else 'N/A'}")
        if rmse < best_score:
            best_score, best_model, best_model_name = rmse, grid_search.best_estimator_, model_name

    print(f"Best model for gas type {gas_type}: {best_model_name} - RMSE: {best_score:.4f}")
    models[gas_type] = best_model
    predictions[gas_type] = {'True': y_val.values, 'Predicted': best_model.predict(X_val)}

# Use different gas models to generate predictions for known concentrations
known_concentrations = np.array([1, 1.5, 2.5, 3.5, 4.5])  # Known concentration values  
all_predictions = {}
model_names = []
rmse_values = []
r2_values = []
gas_values = []
for gas_type, model in models.items():
    subset = data[(data['label'] == gas_type) & (data['Con'].isin(known_concentrations))]
    if subset.empty:
        print(f"Gas type {gas_type} does not have matching known concentration data, skipping prediction.")
        continue
    X_known = subset.drop(columns=['label', 'Con'])
    y_known = subset['Con']

    y_pred_known = model.predict(X_known)
    rmse = np.sqrt(mean_squared_error(y_known, y_pred_known))
    r2 = r2_score(y_known, y_pred_known) if len(y_known) > 1 else None
    print(f"Prediction for gas type {gas_type} - RMSE: {rmse:.4f}, R²: {r2 if r2 is not None else 'N/A'}")

    all_predictions[gas_type] = {
        'Concentration': y_known.values,
        'Predicted': y_pred_known
    }
   
    model_name = model.__class__.__name__ + "_" + str(gas_type)
    model_names.append(model_name)
    rmse_values.append(rmse)
    r2_values.append(r2 if r2 is not None else 0)
    gas_values.append(gas_type)

print(model_names)
print(rmse_values)
print(r2_values)


import matplotlib.pyplot as plt
import numpy as np

# nature_colors = ['#0072B2', '#D55E00', '#F0E442', '#009E73',
#                  '#CC79A7', '#56B4E9', '#E69F00', '#000000',
#                  '#F8766D', '#00BFC4']
nature_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                 '#bcbd22', '#17becf']

plt.figure(figsize=(8, 6))

# Plot a scatter plot of all gases
for idx, (gas_type, preds) in enumerate(all_predictions.items()):
    color = nature_colors[idx % len(nature_colors)]
    true_vals = np.array(preds['Concentration'])
    pred_vals = np.array(preds['Predicted'])
    plt.scatter(true_vals, pred_vals, color=color, alpha=0.7,
                label=f'Gas Type {gas_type}')

# Plot a uniform ideal fit line (y=x) over the range 0 to 5 ppm
x_vals = np.linspace(0.5, 4.8, 100)
plt.plot(x_vals, x_vals, color='red', linestyle='--', linewidth=2,
         label='Ideal Fit')

# Set axis labels, title, and limits    
plt.xlabel('True Concentration (ppm)')
plt.ylabel('Predicted Concentration (ppm)')
# plt.title(f'Predicted vs True Concentration for All Gas Types - {best_model_name}')

plt.xlim(0, 5)
plt.ylim(0, 5)
plt.grid(True)
ticks = np.arange(0, 5, 0.5)
plt.xticks(ticks)
plt.yticks(ticks)
plt.legend(fontsize='small', loc='best')
plt.tight_layout()
plt.show()

# Save the data to a txt file
with open('gas_prediction_results.txt', 'w', encoding='utf-8') as f:
    f.write('Analysis of Gas Concentration Prediction Results\n')
    f.write('=' * 50 + '\n\n')
    
    # Save the prediction results for each gas type
    for gas_type, preds in all_predictions.items():
        f.write(f'Gas Type {gas_type}:\n')
        f.write('-' * 30 + '\n')
        f.write('True Concentration(ppm)\tPredicted Concentration(ppm)\n')
        
        for true_val, pred_val in zip(preds['Concentration'], preds['Predicted']):
            f.write(f'{true_val:.3f}\t\t{pred_val:.3f}\n')
        
        # Calculate and save the statistics for this gas type
        rmse = np.sqrt(mean_squared_error(preds['Concentration'], preds['Predicted']))
        r2 = r2_score(preds['Concentration'], preds['Predicted']) if len(preds['Concentration']) > 1 else 'N/A'
        
        f.write(f'\nStatistics:\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'R²: {r2 if isinstance(r2, str) else r2:.4f}\n')
        f.write('\n' + '=' * 50 + '\n\n')
    
    # Save the average statistics for all gas types
    avg_rmse = np.mean(rmse_values)
    avg_r2 = np.mean([r2 for r2 in r2_values if r2 is not None])
    
    f.write('Overall Statistics:\n')    
    f.write('-' * 30 + '\n')
    f.write(f'Average RMSE: {avg_rmse:.4f}\n')
    f.write(f'Average R²: {avg_r2:.4f}\n\n')
    
    # Save the data points of the ideal fit line (y=x)
    f.write('Ideal Fit Line Data Points:\n')
    f.write('-' * 30 + '\n')
    f.write('X Value(ppm)\tY Value(ppm)\n')
    for x, y in zip(x_vals, x_vals):  # Ideal fit line y=x  
        f.write(f'{x:.3f}\t\t{y:.3f}\n')
    f.write('\n' + '=' * 50 + '\n')

plt.tight_layout()
plt.show()


# Draw a dual Y-axis bar chart
fig, ax1 = plt.subplots(figsize=(12, 6))

# Draw a bar chart of RMSE values
bars = ax1.bar(model_names, rmse_values, color='skyblue', label='RMSE')
ax1.set_xlabel('Models')
ax1.set_ylabel('RMSE')
ax1.tick_params(axis='x', rotation=45)

# Set the range of the RMSE Y-axis
ax1.set_ylim(0, max(rmse_values) + 1.5)  # Add some space at the top

# Use the second Y-axis to draw a line chart of R² values
ax2 = ax1.twinx()
ax2.plot(model_names, r2_values, color='salmon', marker='o', label='R²')
ax2.set_ylabel('R² Score')
ax2.set_ylim(0, max(r2_values) + 0.1)  # Set the range of the R² axis

# Display RMSE values on top of the bars
for bar, rmse in zip(bars, rmse_values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{rmse:.2f}', ha='center', va='bottom', alpha=0.7, fontsize=10)

# Display R² values on top of the line chart
for i, r2 in enumerate(r2_values):
    ax2.text(i, r2, f'{r2:.2f}', ha='center', va='bottom', color='salmon', alpha=0.7, fontsize=10)

# Set the title and display the legend  
plt.title('Comparison of RMSE and R² Across Different Models')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.tight_layout()
plt.show()

# Save the data to a txt file
with open('gas_prediction_results.txt', 'w', encoding='utf-8') as f:
    f.write('Analysis of Gas Concentration Prediction Results\n')
    f.write('=' * 50 + '\n\n')
    
    # Save the prediction results for each gas type
    for gas_type, preds in all_predictions.items():
        f.write(f'Gas Type {gas_type}:\n')
        f.write('-' * 30 + '\n')
        f.write('True Concentration(ppm)\tPredicted Concentration(ppm)\n')
        
        for true_val, pred_val in zip(preds['Concentration'], preds['Predicted']):
            f.write(f'{true_val:.3f}\t\t{pred_val:.3f}\n')
        
        # Calculate and save the statistics for this gas type
        rmse = np.sqrt(mean_squared_error(preds['Concentration'], preds['Predicted']))
        r2 = r2_score(preds['Concentration'], preds['Predicted']) if len(preds['Concentration']) > 1 else 'N/A'
        
        f.write(f'\nStatistics:\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'R²: {r2 if isinstance(r2, str) else r2:.4f}\n')
        f.write('\n' + '=' * 50 + '\n\n')
    
    # Save the average statistics for all gas types
    avg_rmse = np.mean(rmse_values)
    avg_r2 = np.mean([r2 for r2 in r2_values if r2 is not None])
    
    f.write('Overall Statistics:\n')
    f.write('-' * 30 + '\n')
    f.write(f'Average RMSE: {avg_rmse:.4f}\n')
    f.write(f'Average R²: {avg_r2:.4f}\n\n')
    
    # Save the data points of the ideal fit line (y=x)
    f.write('Ideal Fit Line Data Points:\n')
    f.write('-' * 30 + '\n')
    f.write('X Value(ppm)\tY Value(ppm)\n')
    for x, y in zip(x_vals, x_vals):  # Ideal fit line y=x      
        f.write(f'{x:.3f}\t\t{y:.3f}\n')
    f.write('\n' + '=' * 50 + '\n')

plt.tight_layout()
plt.show()

