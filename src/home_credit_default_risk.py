"""
Home Credit Default Risk - Clasificación para Riesgo de Crédito
Competencia de Kaggle: Home Credit Default Risk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Carga los datos principales del dataset"""
    print("📂 Cargando datos...")
    
    try:
        # Cargar datos principales
        app_train = pd.read_csv('application_train.csv')
        app_test = pd.read_csv('application_test.csv')
        
        print(f"✅ Datos cargados exitosamente")
        print(f"📊 application_train shape: {app_train.shape}")
        print(f"📊 application_test shape: {app_test.shape}")
        
        return app_train, app_test
    
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontró el archivo {e.filename}")
        print("📥 Descarga los datos de: https://www.kaggle.com/c/home-credit-default-risk/data")
        return None, None

def explore_data(app_train, app_test):
    """Exploración básica de los datos"""
    print("\n" + "="*60)
    print("🔍 EXPLORACIÓN DE DATOS")
    print("="*60)
    
    # Información básica del dataset de entrenamiento
    print("📈 Información del dataset de entrenamiento:")
    print(f"   - Número de filas: {app_train.shape[0]}")
    print(f"   - Número de columnas: {app_train.shape[1]}")
    print(f"   - Columnas: {list(app_train.columns[:10])}...")  # Primeras 10 columnas
    
    # Variable objetivo
    target_col = 'TARGET'
    if target_col in app_train.columns:
        target_dist = app_train[target_col].value_counts()
        print(f"\n🎯 Distribución de la variable objetivo (TARGET):")
        print(f"   - 0 (No default): {target_dist[0]} ({target_dist[0]/len(app_train)*100:.2f}%)")
        print(f"   - 1 (Default): {target_dist[1]} ({target_dist[1]/len(app_train)*100:.2f}%)")
    
    # Tipos de datos
    print(f"\n📊 Tipos de datos:")
    print(app_train.dtypes.value_counts())
    
    # Valores nulos
    print(f"\n❓ Valores nulos en train:")
    null_counts = app_train.isnull().sum()
    high_null_cols = null_counts[null_counts > len(app_train) * 0.5]  # Columnas con >50% nulos
    print(f"   - Columnas con más del 50% de valores nulos: {len(high_null_cols)}")
    
    return target_col

def preprocess_data(app_train, app_test, target_col):
    """Preprocesamiento de datos para el modelo base"""
    print("\n" + "="*60)
    print("🛠️ PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    # Separar características y variable objetivo
    if target_col in app_train.columns:
        y = app_train[target_col]
        X = app_train.drop(columns=[target_col])
    else:
        print("❌ Error: Columna TARGET no encontrada")
        return None, None, None, None
    
    X_test = app_test.copy()
    
    # 1. Seleccionar solo características numéricas para el modelo base
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"   - Columnas numéricas: {len(numeric_cols)}")
    print(f"   - Columnas categóricas: {len(categorical_cols)}")
    
    # 2. Manejar valores nulos en características numéricas
    imputer = SimpleImputer(strategy='median')
    X_numeric = X[numeric_cols]
    X_test_numeric = X_test[numeric_cols]
    
    X_imputed = pd.DataFrame(imputer.fit_transform(X_numeric), columns=numeric_cols)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_numeric), columns=numeric_cols)
    
    # 3. Codificar variables categóricas simples (solo las que tienen pocas categorías)
    X_processed = X_imputed.copy()
    X_test_processed = X_test_imputed.copy()
    
    # Agregar algunas categóricas importantes si existen
    important_categoricals = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    
    for col in important_categoricals:
        if col in categorical_cols and col in X.columns:
            # Para el modelo base, usar one-hot encoding solo si tiene pocas categorías
            if X[col].nunique() <= 5:
                # Codificar train
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_processed = pd.concat([X_processed, dummies], axis=1)
                
                # Codificar test (asegurando mismas columnas)
                test_dummies = pd.get_dummies(X_test[col], prefix=col, drop_first=True)
                # Alinear columnas
                for dummy_col in dummies.columns:
                    if dummy_col in test_dummies.columns:
                        X_test_processed[dummy_col] = test_dummies[dummy_col]
                    else:
                        X_test_processed[dummy_col] = 0
    
    print(f"   - Características finales: {X_processed.shape[1]}")
    
    # 4. Dividir datos de entrenamiento
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   - Conjunto de entrenamiento: {X_train.shape}")
    print(f"   - Conjunto de validación: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val, X_test_processed, X_processed, y

def create_baseline_model(X_train, X_val, y_train, y_val):
    """Crea y evalúa el modelo base"""
    print("\n" + "="*60)
    print("🚀 CREANDO MODELO BASE")
    print("="*60)
    
    # Modelo 1: Regresión Logística
    print("\n📊 Entrenando Regresión Logística...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Predecir probabilidades para validación
    y_val_pred_lr = lr_model.predict_proba(X_val)[:, 1]
    lr_auc = roc_auc_score(y_val, y_val_pred_lr)
    
    print(f"   ✅ Regresión Logística - AUC: {lr_auc:.4f}")
    
    # Modelo 2: Random Forest
    print("\n📊 Entrenando Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Predecir probabilidades para validación
    y_val_pred_rf = rf_model.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, y_val_pred_rf)
    
    print(f"   ✅ Random Forest - AUC: {rf_auc:.4f}")
    
    # Seleccionar el mejor modelo base
    if rf_auc > lr_auc:
        best_model = rf_model
        best_auc = rf_auc
        best_name = "Random Forest"
    else:
        best_model = lr_model
        best_auc = lr_auc
        best_name = "Logistic Regression"
    
    print(f"\n🏆 MEJOR MODELO BASE: {best_name}")
    print(f"   📊 AUC: {best_auc:.4f}")
    
    return {
        'logistic_regression': {'model': lr_model, 'auc': lr_auc},
        'random_forest': {'model': rf_model, 'auc': rf_auc},
        'best_model': best_model,
        'best_auc': best_auc,
        'best_name': best_name
    }

def create_submission_file(model, X_test_processed, test_ids, filename='submission_baseline.csv'):
    """Crea el archivo de submission para Kaggle"""
    print(f"\n📁 Creando archivo de submission: {filename}")
    
    # Predecir probabilidades para el test set
    test_predictions = model.predict_proba(X_test_processed)[:, 1]
    
    # Crear DataFrame de submission
    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': test_predictions
    })
    
    # Guardar archivo
    submission.to_csv(filename, index=False)
    print(f"✅ Archivo guardado: {filename}")
    print(f"📊 Forma del submission: {submission.shape}")
    print(f"📈 Rango de predicciones: {submission['TARGET'].min():.4f} - {submission['TARGET'].max():.4f}")
    
    return submission

def feature_engineering_experiments(X, y, X_test_processed, test_ids):
    """Experimentos de feature engineering"""
    print("\n" + "="*60)
    print("🔧 EXPERIMENTOS DE FEATURE ENGINEERING")
    print("="*60)
    
    results = {}
    
    # Experimento 1: Modelo base (referencia)
    print("\n🧪 Experimento 1: Modelo Base")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict_proba(X_val)[:, 1]
    auc_base = roc_auc_score(y_val, y_pred_base)
    results['base'] = auc_base
    print(f"   ✅ AUC: {auc_base:.4f}")
    
    # Experimento 2: Más características numéricas
    print("\n🧪 Experimento 2: Características Adicionales")
    # Aquí podrías agregar más características del dataset original
    # Por ahora usamos el mismo set pero con diferentes parámetros
    
    rf_more_trees = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    rf_more_trees.fit(X_train, y_train)
    y_pred_more_trees = rf_more_trees.predict_proba(X_val)[:, 1]
    auc_more_trees = roc_auc_score(y_val, y_pred_more_trees)
    results['more_trees'] = auc_more_trees
    print(f"   ✅ AUC: {auc_more_trees:.4f}")
    
    # Experimento 3: Diferente escalado
    print("\n🧪 Experimento 3: Con Escalado de Características")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    rf_scaled = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = rf_scaled.predict_proba(X_val_scaled)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)
    results['scaled'] = auc_scaled
    print(f"   ✅ AUC: {auc_scaled:.4f}")
    
    # Experimento 4: Balance de clases
    print("\n🧪 Experimento 4: Con Balance de Clases")
    rf_balanced = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        class_weight='balanced'
    )
    rf_balanced.fit(X_train, y_train)
    y_pred_balanced = rf_balanced.predict_proba(X_val)[:, 1]
    auc_balanced = roc_auc_score(y_val, y_pred_balanced)
    results['balanced'] = auc_balanced
    print(f"   ✅ AUC: {auc_balanced:.4f}")
    
    # Experimento 5: Características polinómicas (simuladas)
    print("\n🧪 Experimento 5: Características Interacciones")
    # Para este ejemplo, seleccionamos las 2 características más importantes
    feature_importances = rf_base.feature_importances_
    top_features_idx = np.argsort(feature_importances)[-2:]
    top_features = X.columns[top_features_idx]
    
    if len(top_features) >= 2:
        # Crear característica de interacción
        X_interaction = X.copy()
        interaction_feature = X[top_features[0]] * X[top_features[1]]
        X_interaction[f'{top_features[0]}_x_{top_features[1]}'] = interaction_feature
        
        X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(
            X_interaction, y, test_size=0.2, random_state=42, stratify=y
        )
        
        rf_interaction = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_interaction.fit(X_train_int, y_train_int)
        y_pred_interaction = rf_interaction.predict_proba(X_val_int)[:, 1]
        auc_interaction = roc_auc_score(y_val_int, y_pred_interaction)
        results['interaction'] = auc_interaction
        print(f"   ✅ AUC: {auc_interaction:.4f}")
    
    # Mostrar comparación de resultados
    print("\n" + "="*40)
    print("📊 COMPARACIÓN DE EXPERIMENTOS")
    print("="*40)
    for exp_name, auc_score in results.items():
        print(f"   {exp_name:15} | AUC: {auc_score:.4f}")
    
    # Mejor experimento
    best_exp = max(results, key=results.get)
    print(f"\n🏆 MEJOR EXPERIMENTO: {best_exp}")
    print(f"   📊 AUC: {results[best_exp]:.4f}")
    
    # Crear submission con el mejor modelo
    if best_exp == 'more_trees':
        best_model_exp = rf_more_trees
    elif best_exp == 'balanced':
        best_model_exp = rf_balanced
    elif best_exp == 'interaction':
        best_model_exp = rf_interaction
    else:
        best_model_exp = rf_base
    
    # Entrenar el mejor modelo con todos los datos de entrenamiento
    best_model_exp.fit(X, y)
    
    # Crear submission
    submission_best = create_submission_file(
        best_model_exp, 
        X_test_processed, 
        test_ids,
        f'submission_best_fe_{best_exp}.csv'
    )
    
    return results, best_model_exp

def main():
    """Función principal"""
    print("🏠 HOME CREDIT DEFAULT RISK - CLASIFICACIÓN")
    print("="*60)
    
    # 1. Cargar datos
    app_train, app_test = load_data()
    if app_train is None:
        return
    
    # 2. Exploración de datos
    target_col = explore_data(app_train, app_test)
    
    # 3. Preprocesamiento para modelo base
    result = preprocess_data(app_train, app_test, target_col)
    if result[0] is None:
        return
    X_train, X_val, y_train, y_val, X_test_processed, X_processed, y = result
    
    # 4. Obtener IDs del test set para submission
    test_ids = app_test['SK_ID_CURR'] if 'SK_ID_CURR' in app_test.columns else app_test.index
    
    # 5. Crear modelo base
    baseline_results = create_baseline_model(X_train, X_val, y_train, y_val)
    
    # 6. Crear submission del modelo base
    submission_baseline = create_submission_file(
        baseline_results['best_model'], 
        X_test_processed, 
        test_ids
    )
    
    # 7. Experimentos de feature engineering
    feature_results, best_fe_model = feature_engineering_experiments(
        X_processed, y, X_test_processed, test_ids
    )
    
    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60)
    print("📁 Archivos de submission creados:")
    print("   - submission_baseline.csv (Modelo base)")
    print("   - submission_best_fe_*.csv (Mejor feature engineering)")
    print("\n📊 Métrica de evaluación: AUC-ROC")
   

if __name__ == "__main__":
    main()
