import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna

# Загрузка данных из CSV-файла
st.title("Оценка точности с использованием CatBoostClassifier и Optuna")

uploaded_file = st.file_uploader("Загрузите свой CSV-файл с данными", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Определение целевой переменной
    st.subheader("Выберите целевую переменную:")
    target_col = st.selectbox("Целевая переменная (Target)", df.columns)

    # Выбор признаков и целевой переменной
    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

    # Оптимизация CatBoostClassifier с использованием Optuna
    st.subheader("Оптимизация CatBoostClassifier:")
    params = {
        'iterations': st.slider("Количество итераций", 10, 100, 50),
        'depth': st.slider("Глубина", 1, 10, 6),
        'learning_rate': st.slider("Скорость обучения", 0.001, 0.1, 0.01),
        'random_seed': 0,
        'verbose': 0
    }

    def objective(trial):
        params['iterations'] = trial.suggest_int('iterations', 10, 100)
        params['depth'] = trial.suggest_int('depth', 1, 10)
        params['learning_rate'] = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    best_accuracy = study.best_value

    # Обучение модели с лучшими параметрами
    st.subheader("Обучение модели с лучшими параметрами:")
    best_model = CatBoostClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Прогнозирование и оценка точности
    st.subheader("Оценка точности модели:")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy Score: {accuracy:.2f}")
    st.write(f"Лучшие параметры модели: {best_params}")

    # Отображение результатов
    st.subheader("Результаты:")
    st.write("Модель обучена и оценена на вашем наборе данных.")