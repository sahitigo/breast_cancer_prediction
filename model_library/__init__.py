import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

def classification_models(X, y, max_depth=None, leaf_nodes=1, n_neighbors=5):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    train_df = scaler.fit_transform(X_train)
    test_df = scaler.transform(X_test)
    
    # Convert the scaled arrays back to DataFrames
    X_train = pd.DataFrame(train_df, columns=X_train.columns)
    X_test = pd.DataFrame(test_df, columns=X_train.columns)
    
    # Train logistic regression model
    print("Training Logistic Regression...")
    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)
    
    # Make predictions for logistic regression
    log_train_predictions = model_lr.predict(X_train)
    log_test_predictions = model_lr.predict(X_test)
    
    # Calculate accuracy scores for logistic regression
    log_train_accuracy = accuracy_score(y_train, log_train_predictions)
    log_test_accuracy = accuracy_score(y_test, log_test_predictions)
  
    # Calculate additional metrics for logistic regression
    log_train_f1 = f1_score(y_train, log_train_predictions)
    log_test_f1 = f1_score(y_test, log_test_predictions)
    log_train_sensitivity = recall_score(y_train, log_train_predictions)
    log_test_sensitivity = recall_score(y_test, log_test_predictions)
    
    print()
    
    # Train KNN classification model
    print("Training KNN Classification...")
    model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    model_knn.fit(X_train, y_train)
    
    # Make predictions for KNN classification
    knn_train_predictions = model_knn.predict(X_train)
    knn_test_predictions = model_knn.predict(X_test)
    
    # Calculate accuracy scores for KNN classification
    knn_train_accuracy = accuracy_score(y_train, knn_train_predictions)
    knn_test_accuracy = accuracy_score(y_test, knn_test_predictions)
    
    # Calculate additional metrics for KNN classification
    knn_train_f1 = f1_score(y_train, knn_train_predictions)
    knn_test_f1 = f1_score(y_test, knn_test_predictions)
    knn_train_sensitivity = recall_score(y_train, knn_train_predictions)
    knn_test_sensitivity = recall_score(y_test, knn_test_predictions)
    
    print()
    
    # Train Decision Tree classification model
    print("Training Decision Tree Classification...")
    model_dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=leaf_nodes)
    model_dt.fit(X_train, y_train)
    
    # Make predictions for Decision Tree classification
    dt_train_predictions = model_dt.predict(X_train)
    dt_test_predictions = model_dt.predict(X_test)
    
    # Calculate accuracy scores for Decision Tree classification
    dt_train_accuracy = accuracy_score(y_train, dt_train_predictions)
    dt_test_accuracy = accuracy_score(y_test, dt_test_predictions)
    
    # Calculate additional metrics for Decision Tree classification
    dt_train_f1 = f1_score(y_train, dt_train_predictions)
    dt_test_f1 = f1_score(y_test, dt_test_predictions)
    dt_train_sensitivity = recall_score(y_train, dt_train_predictions)
    dt_test_sensitivity = recall_score(y_test, dt_test_predictions)
    
    print()
    
    model_performance = pd.DataFrame({
        'Name': ['Logistic Regression', 'KNN Classification', 'Decision Tree Classification'],
        'Train Accuracy': [log_train_accuracy, knn_train_accuracy, dt_train_accuracy],
        'Test Accuracy': [log_test_accuracy, knn_test_accuracy, dt_test_accuracy],
        'Train F1 Score': [log_train_f1, knn_train_f1, dt_train_f1],
        'Test F1 Score': [log_test_f1, knn_test_f1, dt_test_f1],
        'Train Sensitivity': [log_train_sensitivity, knn_train_sensitivity, dt_train_sensitivity],
        'Test Sensitivity': [log_test_sensitivity, knn_test_sensitivity, dt_test_sensitivity]
    })
    
    return model_performance
