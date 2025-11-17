
import numpy as np
import pandas as pd
import random
from datetime import datetime
import MetaTrader5 as mt5
import time
import threading
import queue
from typing import Optional, Tuple, List, Dict, Any
from sklearn.utils import class_weight
from imblearn.under_sampling import RandomUnderSampler
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# GLOBALS
# Новые глобальные переменные для балансировки риска
TOTAL_PORTFOLIO_RISK: float = 1000.0  # Общий риск на весь портфель
POSITION_SIZES: Dict[str, float] = {}  # Размеры позиций по инструментам
SYMBOL_TRADES: Dict[str, bool] = {}    # Отслеживание сделок (одна на инструмент)
MARKUP: float = 0.000001
BACKWARD: datetime = datetime(2021, 1, 1)
FORWARD: datetime = datetime(2024, 1, 1)
EXAMWARD: datetime = datetime(2024, 7, 1)
MAX_OPEN_TRADES: int = 6
SYMBOL: str = "EURUSD"
RISK_REWARD_RATIO: int = 4
TERMINAL_PATH: str = r"C:\Program Files\RoboForex MT5 Terminal\terminal64.exe"

# Очередь для логов с ограничением размера
log_queue = queue.Queue(maxsize=1000)

def log_printer():
    while True:
        try:
            log_message = log_queue.get(timeout=10)
            if log_message is None:
                break
            print(log_message)
        except queue.Empty:
            continue

# Запускаем поток для печати логов
printer_thread = threading.Thread(target=log_printer, daemon=True)
printer_thread.start()

def log(message: str) -> None:
    try:
        log_queue.put(message, block=False)
    except queue.Full:
        pass

def retrieve_data(symbol: str, retries_limit: int = 300) -> Optional[pd.DataFrame]:
    attempt: int = 0
    while attempt < retries_limit:
        if not mt5.initialize(path=TERMINAL_PATH):
            log("Terminal initialization error")
            return None

        instrument_count: int = mt5.symbols_total()
        if instrument_count > 0:
            log(f"Instruments in terminal: {instrument_count}")
        else:
            log("No instruments in terminal")

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, BACKWARD, EXAMWARD)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            log(f"No data for symbol {symbol} yet (attempt {attempt + 1})")
            attempt += 1
            time.sleep(1)
            continue

        raw_data = pd.DataFrame(
            rates[:-1],
            columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'],
            dtype=np.float64  # Изменено на float64 для GMM
        )
        raw_data['time'] = pd.to_datetime(raw_data['time'], unit='s')
        raw_data.set_index('time', inplace=True)

        # ОРИГИНАЛЬНЫЕ ПРИЗНАКИ (сохранены как есть)
        raw_data['raw_SMA_10'] = raw_data['close'].rolling(window=10).mean().astype(np.float64)
        raw_data['raw_SMA_20'] = raw_data['close'].rolling(window=20).mean().astype(np.float64)
        raw_data['Price_Change'] = raw_data['close'].pct_change().mul(100).astype(np.float64)
        raw_data['raw_Std_Dev_Close'] = raw_data['close'].rolling(window=20).std().astype(np.float64)
        raw_data['raw_Volume_Change'] = raw_data['tick_volume'].pct_change().mul(100).astype(np.float64)
        raw_data['raw_Prev_Day_Price_Change'] = (raw_data['close'] - raw_data['close'].shift(1)).astype(np.float64)
        raw_data['raw_Prev_Week_Price_Change'] = (raw_data['close'] - raw_data['close'].shift(7)).astype(np.float64)
        raw_data['raw_Prev_Month_Price_Change'] = (raw_data['close'] - raw_data['close'].shift(30)).astype(np.float64)

        raw_data['Price_Volume_Ratio'] = np.where(
            raw_data['tick_volume'] != 0,
            raw_data['close'] / raw_data['tick_volume'],
            0
        ).astype(np.float64)

        raw_data['Consecutive_Positive_Changes'] = (
            (raw_data['Price_Change'] > 0).astype(np.int8)
            .groupby((raw_data['Price_Change'] > 0).astype(np.int8).diff().ne(0).cumsum())
            .cumsum()
        ).replace([np.inf, -np.inf, np.nan], 0).astype(np.int16)
        raw_data['Consecutive_Negative_Changes'] = (
            (raw_data['Price_Change'] < 0).astype(np.int8)
            .groupby((raw_data['Price_Change'] < 0).astype(np.int8).diff().ne(0).cumsum())
            .cumsum()
        ).replace([np.inf, -np.inf, np.nan], 0).astype(np.int16)
        raw_data['Price_Density'] = raw_data['close'].rolling(window=10).apply(
            lambda x: len(set(x)), raw=True
        ).replace([np.inf, -np.inf, np.nan], 0).astype(np.int16)
        raw_data['Fractal_Analysis'] = raw_data['close'].rolling(window=10).apply(
            lambda x: 1 if x.argmax() == len(x) - 1 else (-1 if x.argmin() == len(x) - 1 else 0), raw=True
        ).replace([np.inf, -np.inf, np.nan], 0).astype(np.int8)
        raw_data['Median_Close_7'] = raw_data['close'].rolling(window=7).median().astype(np.float64)
        raw_data['Median_Close_30'] = raw_data['close'].rolling(window=30).median().astype(np.float64)
        raw_data['Price_Volatility'] = (
            raw_data['close'].rolling(window=20).std() / raw_data['close'].rolling(window=20).mean()
        ).replace([np.inf, -np.inf, np.nan], 0).astype(np.float64)

        # ===== НОВЫЕ ОБОГАЩАЮЩИЕ ПРИЗНАКИ =====
        
        # 1. RSI (Relative Strength Index) - классический осциллятор
        delta = raw_data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        raw_data['RSI_14'] = (100 - (100 / (1 + rs))).astype(np.float64)
        
        # 2-3. MACD (Moving Average Convergence Divergence)
        ema12 = raw_data['close'].ewm(span=12).mean()
        ema26 = raw_data['close'].ewm(span=26).mean()
        raw_data['MACD_Line'] = (ema12 - ema26).astype(np.float64)
        raw_data['MACD_Signal'] = raw_data['MACD_Line'].ewm(span=9).mean().astype(np.float64)
        
        # 4. ATR (Average True Range) - волатильность
        tr1 = raw_data['high'] - raw_data['low']
        tr2 = abs(raw_data['high'] - raw_data['close'].shift(1))
        tr3 = abs(raw_data['low'] - raw_data['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        raw_data['ATR_14'] = true_range.rolling(14).mean().astype(np.float64)
        
        # 5. Bollinger Bands Position
        bb_middle = raw_data['close'].rolling(20).mean()
        bb_std = raw_data['close'].rolling(20).std()
        raw_data['BB_Position'] = ((raw_data['close'] - bb_middle) / (2 * bb_std)).astype(np.float64)
        
        # 6-7. Stochastic Oscillator
        low_14 = raw_data['low'].rolling(14).min()
        high_14 = raw_data['high'].rolling(14).max()
        raw_data['Stoch_K'] = (100 * (raw_data['close'] - low_14) / (high_14 - low_14)).astype(np.float64)
        raw_data['Stoch_D'] = raw_data['Stoch_K'].rolling(3).mean().astype(np.float64)
        
        # 8. Williams %R
        raw_data['Williams_R'] = (-100 * (high_14 - raw_data['close']) / (high_14 - low_14)).astype(np.float64)
        
        # 9. CCI (Commodity Channel Index)
        typical_price = (raw_data['high'] + raw_data['low'] + raw_data['close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad_tp = typical_price.rolling(20).apply(lambda x: abs(x - x.mean()).mean())
        raw_data['CCI_20'] = ((typical_price - sma_tp) / (0.015 * mad_tp)).astype(np.float64)
        
        # 10. Momentum
        raw_data['Momentum_10'] = (raw_data['close'] / raw_data['close'].shift(10) * 100).astype(np.float64)
        
        # 11. Rate of Change (ROC)
        raw_data['ROC_12'] = (raw_data['close'].pct_change(12) * 100).astype(np.float64)
        
        # 12-13. Объемные индикаторы
        raw_data['Volume_SMA_Ratio'] = (raw_data['tick_volume'] / raw_data['tick_volume'].rolling(20).mean()).astype(np.float64)
        raw_data['Price_Volume_Trend'] = ((raw_data['close'].pct_change() * raw_data['tick_volume']).rolling(10).sum()).astype(np.float64)
        
        # 14-15. Временные циклические признаки
        raw_data['Hour_Sin'] = np.sin(2 * np.pi * raw_data.index.hour / 24).astype(np.float64)
        raw_data['DayOfWeek_Sin'] = np.sin(2 * np.pi * raw_data.index.dayofweek / 7).astype(np.float64)
        
        # 16-17. Лаговые признаки (краткосрочная память)
        raw_data['Close_Lag_1'] = raw_data['close'].shift(1).astype(np.float64)
        raw_data['Close_Lag_5'] = raw_data['close'].shift(5).astype(np.float64)
        
        # 18. Улучшенный фрактальный анализ
        raw_data['Fractal_Enhanced'] = raw_data['close'].rolling(window=5).apply(
            lambda x: 1 if x.argmax() == 2 else (-1 if x.argmin() == 2 else 0), raw=True
        ).astype(np.float64)
        
        # 19. Keltner Channel Position
        kc_middle = raw_data['close'].ewm(span=20).mean()
        kc_range = true_range.ewm(span=20).mean() * 2
        raw_data['Keltner_Position'] = ((raw_data['close'] - kc_middle) / kc_range).astype(np.float64)
        
        # 20. Efficiency Ratio (адаптивность)
        direction = abs(raw_data['close'] - raw_data['close'].shift(10))
        volatility = raw_data['close'].diff().abs().rolling(10).sum()
        efficiency_ratio = direction / volatility
        raw_data['Efficiency_Ratio'] = efficiency_ratio.astype(np.float64)
        
        # 21. Money Flow Index (MFI) - объемный RSI
        typical_price_mfi = (raw_data['high'] + raw_data['low'] + raw_data['close']) / 3
        money_flow = typical_price_mfi * raw_data['tick_volume']
        positive_flow = money_flow.where(typical_price_mfi > typical_price_mfi.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price_mfi < typical_price_mfi.shift(1), 0).rolling(14).sum()
        money_ratio = positive_flow / negative_flow
        raw_data['MFI_14'] = (100 - (100 / (1 + money_ratio))).astype(np.float64)
        
        # 22. TRIX (тройное экспоненциальное сглаживание)
        ema1 = raw_data['close'].ewm(span=14).mean()
        ema2 = ema1.ewm(span=14).mean()
        ema3 = ema2.ewm(span=14).mean()
        raw_data['TRIX'] = (ema3.pct_change() * 10000).astype(np.float64)
        
        # 23. Aroon индикатор
        aroon_up = (14 - raw_data['high'].rolling(14).apply(lambda x: 14 - 1 - x.argmax())) * 100 / 14
        aroon_down = (14 - raw_data['low'].rolling(14).apply(lambda x: 14 - 1 - x.argmin())) * 100 / 14
        raw_data['Aroon_Oscillator'] = (aroon_up - aroon_down).astype(np.float64)
        
        # 24. Ultimate Oscillator компонент
        bp = raw_data['close'] - pd.concat([raw_data['low'], raw_data['close'].shift(1)], axis=1).min(axis=1)
        tr_uo = pd.concat([raw_data['high'] - raw_data['low'], 
                          abs(raw_data['high'] - raw_data['close'].shift(1)),
                          abs(raw_data['low'] - raw_data['close'].shift(1))], axis=1).max(axis=1)
        raw_data['UO_Component'] = (bp.rolling(7).sum() / tr_uo.rolling(7).sum()).astype(np.float64)
        
        # 25. Chaikin Oscillator (объемный индикатор)
        ad_line = ((raw_data['close'] - raw_data['low']) - (raw_data['high'] - raw_data['close'])) / (raw_data['high'] - raw_data['low']) * raw_data['tick_volume']
        ad_line = ad_line.fillna(0).cumsum()
        raw_data['Chaikin_Osc'] = (ad_line.ewm(span=3).mean() - ad_line.ewm(span=10).mean()).astype(np.float64)

        # Финальная очистка данных
        raw_data = raw_data.replace([np.inf, -np.inf], np.nan)
        
        # Более аккуратная обработка NaN
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if raw_data[col].isna().any():
                # Для большинства колонок используем forward fill, затем backward fill
                raw_data[col] = raw_data[col].fillna(method='ffill').fillna(method='bfill')
                # Если всё ещё есть NaN, заполняем медианой
                if raw_data[col].isna().any():
                    raw_data[col] = raw_data[col].fillna(raw_data[col].median())

        log("\nOriginal columns:")
        log(str(raw_data[['close', 'high', 'low', 'open', 'tick_volume']].tail(10)))
        log(f"\nTotal features: {len(raw_data.columns)}")
        log("\nList of features:")
        log(str(raw_data.columns.tolist()))
        log("\nLast 10 features:")
        log(str(raw_data.tail(10)))

        return raw_data

    log(f"Failed after {retries_limit} attempts to retrieve data")
    return None

def augment_data(
    raw_data: pd.DataFrame,
    noise_level: float = 0.01,
    time_shift: int = 1,
    scale_range: Tuple[float, float] = (0.9, 1.1)
) -> pd.DataFrame:
    log(f"Rows before augmentation: {len(raw_data)}")
    augmented_data = [raw_data]

    noisy_data = raw_data.astype(np.float32) + np.random.normal(0, noise_level, raw_data.shape).astype(np.float32)
    noisy_data = noisy_data.replace([np.inf, -np.inf], np.nan).fillna(raw_data.median(numeric_only=True))
    augmented_data.append(noisy_data)
    log(f"Added {len(noisy_data)} rows after adding noise")

    shifted_data = raw_data.copy()
    shifted_data.index += pd.DateOffset(hours=time_shift)
    shifted_data = shifted_data.replace([np.inf, -np.inf], np.nan).fillna(raw_data.median(numeric_only=True))
    augmented_data.append(shifted_data)
    log(f"Added {len(shifted_data)} rows after time shifting")

    scale = np.random.uniform(scale_range[0], scale_range[1])
    scaled_data = raw_data.astype(np.float32) * scale
    scaled_data = scaled_data.replace([np.inf, -np.inf], np.nan).fillna(raw_data.median(numeric_only=True))
    augmented_data.append(scaled_data)
    log(f"Added {len(scaled_data)} rows after scaling")

    inverted_data = raw_data.copy()
    price_columns = ['open', 'high', 'low', 'close', 'raw_SMA_10', 'raw_SMA_20', 
                     'raw_Std_Dev_Close', 'raw_Prev_Day_Price_Change', 
                     'raw_Prev_Week_Price_Change', 'raw_Prev_Month_Price_Change', 
                     'Price_Volatility', 'Median_Close_7', 'Median_Close_30']
    for col in price_columns:
        if col in inverted_data.columns:
            inverted_data[col] *= -1
    inverted_data = inverted_data.replace([np.inf, -np.inf], np.nan).fillna(raw_data.median(numeric_only=True))
    augmented_data.append(inverted_data)
    log(f"Added {len(inverted_data)} rows after inversion")

    result = pd.concat(augmented_data, ignore_index=False)
    log(f"Rows after augmentation: {len(result)}")

    log("Print dates by years:")
    for year, group in result.groupby(result.index.year):
        log(f"Year {year}: {len(group)} rows")

    if 'tick_volume' in result.columns:
        result['tick_volume'] = result['tick_volume'].clip(lower=0).astype(np.float32)

    result = result.replace([np.inf, -np.inf], np.nan).fillna(result.median(numeric_only=True))

    del augmented_data, noisy_data, shifted_data, scaled_data, inverted_data
    return result

def markup_data(
    data: pd.DataFrame,
    target_column: str,
    label_column: str,
    markup_ratio: float = 0.00002
) -> pd.DataFrame:
    log("Starting markup_data function")
    data[label_column] = np.where(
        data[target_column].shift(-1) > data[target_column] + markup_ratio,
        1,
        0
    ).astype(np.int8)
    data.loc[data[label_column].isna(), label_column] = 0
    log(f"Number of labels set for price change greater than markup ratio: {data[label_column].sum()}")
    return data

def label_data(
    data: pd.DataFrame,
    symbol: str,
    min_days: int = 2,
    max_days: int = 72
) -> pd.DataFrame:
    if not mt5.initialize(path=TERMINAL_PATH):
        log("Terminal connection error")
        return data

    symbol_info = mt5.symbol_info(symbol)
    stop_level: float = 300 * symbol_info.point
    take_level: float = 800 * symbol_info.point
    labels: List[Optional[int]] = []

    for i in range(data.shape[0] - max_days):
        rand: int = random.randint(min_days, max_days)
        curr_pr: float = data['close'].iloc[i]
        future_pr: float = data['close'].iloc[i + rand]
        min_pr: float = data['low'].iloc[i:i + rand].min()
        max_pr: float = data['high'].iloc[i:i + rand].max()
        price_change: float = abs(future_pr - curr_pr)

        if (price_change > take_level and
            future_pr > curr_pr and
            min_pr > curr_pr - stop_level):
            labels.append(1)
        elif (price_change > take_level and
              future_pr < curr_pr and
              max_pr < curr_pr + stop_level):
            labels.append(0)
        else:
            labels.append(None)

    data = data.iloc[:len(labels)].copy()
    data['labels'] = labels
    data.dropna(inplace=True)

    X = data.drop('labels', axis=1)
    y = data['labels']

    rus = RandomUnderSampler(random_state=2)
    X_balanced, y_balanced = rus.fit_resample(X, y)

    data_balanced = pd.concat([X_balanced, y_balanced], axis=1)
    log(f"Number of growth labels (1.0): {data_balanced['labels'].value_counts().get(1.0, 0)}")
    log(f"Number of decline labels (0.0): {data_balanced['labels'].value_counts().get(0.0, 0)}")

    return data_balanced

def generate_new_features(
    data: pd.DataFrame,
    num_features: int = 10,
    random_seed: int = 1
) -> pd.DataFrame:
    random.seed(random_seed)
    new_features: Dict[str, pd.Series] = {}

    columns = data.columns
    for i in range(num_features):
        feature_name = f'feature_{i}'
        col1_idx, col2_idx = random.sample(range(len(columns)), 2)
        col1, col2 = columns[col1_idx], columns[col2_idx]
        operation = random.choice([
            'add', 'subtract', 'multiply', 'divide',
            'shift', 'rolling_mean', 'rolling_std',
            'rolling_max', 'rolling_min', 'rolling_sum'
        ])

        if operation == 'add':
            new_features[feature_name] = (data[col1] + data[col2]).astype(np.float32)
        elif operation == 'subtract':
            new_features[feature_name] = (data[col1] - data[col2]).astype(np.float32)
        elif operation == 'multiply':
            new_features[feature_name] = (data[col1] * data[col2]).astype(np.float32)
        elif operation == 'divide':
            new_features[feature_name] = (data[col1] / (data[col2] + 1e-8)).astype(np.float32)
        elif operation == 'shift':
            shift = random.randint(1, 10)
            new_features[feature_name] = data[col1].shift(shift).fillna(method='ffill').fillna(method='bfill').astype(np.float32)
        elif operation == 'rolling_mean':
            window = random.randint(2, 20)
            new_features[feature_name] = data[col1].rolling(window).mean().astype(np.float32)
        elif operation == 'rolling_std':
            window = random.randint(2, 20)
            new_features[feature_name] = data[col1].rolling(window).std().astype(np.float32)
        elif operation == 'rolling_max':
            window = random.randint(2, 20)
            new_features[feature_name] = data[col1].rolling(window).max().astype(np.float32)
        elif operation == 'rolling_min':
            window = random.randint(2, 20)
            new_features[feature_name] = data[col1].rolling(window).min().astype(np.float32)
        elif operation == 'rolling_sum':
            window = random.randint(2, 20)
            new_features[feature_name] = data[col1].rolling(window).sum().astype(np.float32)

    new_data = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
    new_data = new_data.replace([np.inf, -np.inf], np.nan).fillna(new_data.median(numeric_only=True))
    log("\nGenerated features:")
    log(str(new_data[list(new_features.keys())].tail(100)))
    return new_data

def cluster_features_by_gmm(
    data: pd.DataFrame,
    n_components: int = 6
) -> pd.DataFrame:
    X = data.drop(['label', 'labels'], axis=1, errors='ignore').astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        reg_covar=0.1,
        random_state=1
    )
    gmm.fit(X)
    data['cluster'] = gmm.predict(X).astype(np.int16)
    log("\nFeature clusters:")
    log(str(data[['cluster']].tail(100)))
    return data

def feature_engineering(
    data: pd.DataFrame,
    n_features_to_select: int = 15
) -> pd.DataFrame:
    X = data.drop(['label', 'labels'], axis=1, errors='ignore').astype(np.float32)
    y = data['labels'].astype(np.int8)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))

    unique_classes = y.unique()
    if len(unique_classes) < 2:
        log(f"Error in feature_engineering: Only one class found in labels: {unique_classes}")
        raise ValueError(f"The target 'y' needs to have more than 1 class. Got {len(unique_classes)} class instead")

    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=5,
        scoring='accuracy',
        n_jobs=1,
        verbose=1,
        min_features_to_select=n_features_to_select
    )
    rfecv.fit(X, y)

    selected_features = X.columns[rfecv.get_support()]
    selected_data = data[selected_features.tolist() + ['label', 'labels']]
    log("\nBest features:")
    log(str(pd.DataFrame({'Feature': selected_features})))
    return selected_data

def train_xgboost_classifier(
    data: pd.DataFrame,
    num_boost_rounds: int = 500
) -> BaggingClassifier:
    if data.empty:
        raise ValueError("Data should not be empty")

    required_columns = ['label', 'labels']
    if not all(column in data.columns for column in required_columns):
        raise ValueError(f"Data is missing required columns: {required_columns}")

    X = data.drop(['label', 'labels'], axis=1).astype(np.float32)
    y = data['labels'].astype(np.int8)

    if not all(pd.api.types.is_numeric_dtype(X[column]) for column in X.columns):
        raise ValueError("All features should have numeric data type")

    unique_classes = y.unique()
    if len(unique_classes) < 2:
        log(f"Error in train_xgboost_classifier: Only one class found in labels: {unique_classes}")
        raise ValueError(f"The target 'y' needs to have more than 1 class. Got {len(unique_classes)} class instead")

    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=1,
        max_depth=5,
        learning_rate=0.2,
        n_estimators=300,
        subsample=0.01,
        colsample_bytree=0.1,
        reg_alpha=1,
        reg_lambda=1
    )

    bagging_clf = BaggingClassifier(estimator=clf, random_state=1)
    param_grid = {
        'n_estimators': [10, 20, 30],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0]
    }

    grid_search = GridSearchCV(bagging_clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    accuracy = grid_search.best_score_
    log(f"Average cross-validation accuracy: {accuracy:.2f}")
    return grid_search.best_estimator_

def test_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    markup: float,
    initial_balance: float = 10000.0,
    point_cost: float = 0.00001
) -> None:
    balance = initial_balance
    trades = 0
    profits = []

    predicted_labels = model.predict(X_test.values)
    close = X_test['close'].values

    for i in range(len(predicted_labels) - 10):
        entry_price = close[i]
        exit_price = close[i + 10]
        if predicted_labels[i] == 1:
            if exit_price > entry_price + markup:
                profit = (exit_price - entry_price - markup) / point_cost
                balance += profit
                trades += 1
                profits.append(profit)
            else:
                loss = (entry_price - exit_price + markup) / point_cost
                balance -= loss
                trades += 1
                profits.append(-loss)
        elif predicted_labels[i] == 0:
            if exit_price < entry_price - markup:
                profit = (entry_price - exit_price - markup) / point_cost
                balance += profit
                trades += 1
                profits.append(profit)
            else:
                loss = (exit_price - entry_price + markup) / point_cost
                balance -= loss
                trades += 1
                profits.append(-loss)

    total_profit = balance - initial_balance
    log(f"Total accumulated profit or loss: {total_profit:.2f}")
    log(f"Number of trades: {trades}")
    time.sleep(100)


def calculate_portfolio_position_sizes(symbols: List[str]) -> None:
    """Расчет размеров позиций для всего портфеля на основе фиксированного стоп-лосса"""
    global POSITION_SIZES
    
    if not mt5.initialize(path=TERMINAL_PATH):
        log("Ошибка подключения к терминалу для расчета позиций")
        return
    
    try:
        # Риск на инструмент (равномерное распределение)
        risk_per_instrument = TOTAL_PORTFOLIO_RISK / len(symbols)
        log(f"Общий риск портфеля: {TOTAL_PORTFOLIO_RISK} USD")
        log(f"Количество инструментов: {len(symbols)}")
        log(f"Риск на инструмент: {risk_per_instrument:.2f} USD")
        
        # Рассчитываем размеры позиций для каждого символа
        for symbol in symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                log(f"Не удалось получить информацию о символе {symbol}")
                continue
            
            # Выводим детальную информацию о символе
            log(f"\n=== Анализ символа {symbol} ===")
            log(f"Point: {symbol_info.point}")
            log(f"Trade tick value: {symbol_info.trade_tick_value}")
            log(f"Volume min: {symbol_info.volume_min}")
            log(f"Volume step: {symbol_info.volume_step}")
            log(f"Volume max: {symbol_info.volume_max}")
            
            # Фиксированный стоп-лосс 300 пипсов
            stop_loss_pips = 300
            
            # Риск на стандартный лот = стоп-лосс в пипсах * стоимость пипса
            risk_per_lot = stop_loss_pips * symbol_info.trade_tick_value
            log(f"Риск на 1 лот при SL 300 пипсов: {risk_per_lot:.2f} USD")
            
            # Размер позиции = допустимый риск / риск на лот
            base_size = risk_per_instrument / risk_per_lot
            log(f"Расчетный размер позиции: {base_size:.6f} лот")
            
            # Нормализуем под требования брокера
            min_lot = symbol_info.volume_min
            lot_step = symbol_info.volume_step
            normalized_size = max(min_lot, round(base_size / lot_step) * lot_step)
            
            POSITION_SIZES[symbol] = normalized_size
            SYMBOL_TRADES[symbol] = False  # Изначально сделок нет
            
            # Пересчитываем итоговый риск
            actual_risk = normalized_size * risk_per_lot
            log(f"Нормализованный размер: {normalized_size:.3f} лот")
            log(f"ИТОГОВЫЙ РИСК: {actual_risk:.2f} USD")
            log(f"Отклонение от целевого риска: {actual_risk - risk_per_instrument:.2f} USD")
        
        # Проверяем, что все символы получили размеры позиций
        log(f"\n=== ФИНАЛЬНАЯ ПРОВЕРКА ===")
        total_calculated_risk = 0
        for symbol in symbols:
            if symbol not in POSITION_SIZES:
                log(f"ОШИБКА: Не удалось рассчитать размер позиции для {symbol}")
            else:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    risk = POSITION_SIZES[symbol] * 300 * symbol_info.trade_tick_value
                    total_calculated_risk += risk
                    log(f"{symbol}: {POSITION_SIZES[symbol]:.3f} лот, риск: {risk:.2f} USD")
        
        log(f"\nОбщий рассчитанный риск портфеля: {total_calculated_risk:.2f} USD")
        log(f"Целевой риск портфеля: {TOTAL_PORTFOLIO_RISK} USD")
        log(f"Отклонение: {abs(total_calculated_risk - TOTAL_PORTFOLIO_RISK):.2f} USD")
    
    except Exception as e:
        log(f"Ошибка расчета позиций: {e}")
        import traceback
        log(f"Детали ошибки: {traceback.format_exc()}")
    finally:
        mt5.shutdown()
        
def check_and_update_position_flags() -> None:
    """Проверяет открытые позиции и сбрасывает флаги для закрытых позиций"""
    global SYMBOL_TRADES
    
    if not mt5.initialize(path=TERMINAL_PATH):
        return
    
    try:
        # Получаем все открытые позиции
        positions = mt5.positions_get()
        if positions is None:
            positions = []
        
        # Получаем список символов с открытыми позициями
        open_symbols = set()
        for pos in positions:
            if pos.magic == 123456:  # Наш magic number
                open_symbols.add(pos.symbol)
        
        # Сбрасываем флаги для символов без открытых позиций
        for symbol in list(SYMBOL_TRADES.keys()):
            if symbol not in open_symbols and SYMBOL_TRADES[symbol]:
                SYMBOL_TRADES[symbol] = False
                log(f"Флаг сброшен для {symbol} - позиция закрыта")
    
    except Exception as e:
        log(f"Ошибка проверки позиций: {e}")
    finally:
        mt5.shutdown()
        
def online_trading(
    symbol: str,
    features: np.ndarray,
    model: Any
) -> Optional[Any]:
    global SYMBOL_TRADES, POSITION_SIZES
    
    # Проверяем и обновляем флаги позиций
    check_and_update_position_flags()
    
    if not mt5.initialize(path=TERMINAL_PATH):
        log("Error: Failed to connect to MetaTrader 5 terminal")
        return None

    # Проверяем, открыта ли уже позиция по этому символу
    if SYMBOL_TRADES.get(symbol, False):
        log(f"Позиция по {symbol} уже открыта")
        mt5.shutdown()
        return None

    # Получаем размер позиции из расчета портфеля
    if symbol not in POSITION_SIZES:
        log(f"Error: Position size not calculated for {symbol}")
        mt5.shutdown()
        return None
    
    volume = POSITION_SIZES[symbol]
    
    attempts: int = 30000
    account_info = mt5.account_info()
    account_balance: float = account_info.balance

    symbol_info = None
    for _ in range(attempts):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is not None:
            break
        log(f"Error: Instrument not found. Attempt {_ + 1} of {attempts}")
        time.sleep(5)

    if symbol_info is None:
        mt5.shutdown()
        return None

    tick = mt5.symbol_info_tick(symbol)
    price_bid: float = tick.bid
    price_ask: float = tick.ask
    
    # НОВАЯ ПРОВЕРКА СПРЕДА - НЕ БОЛЕЕ 35 ПИПСОВ
    spread_pips = (price_ask - price_bid) / symbol_info.point
    if spread_pips > 35:
        log(f"Спред слишком большой для {symbol}: {spread_pips:.1f} пипсов (максимум 35)")
        mt5.shutdown()
        return None
    
    signal = model.predict(features)
    positions_total: int = mt5.positions_total()

    request = None
    if positions_total < MAX_OPEN_TRADES and signal[-1] > 0.5:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price_ask,
            "sl": price_ask - 350 * symbol_info.point,
            "tp": price_ask + 800 * symbol_info.point,
            "deviation": 20,
            "magic": 123456,
            "comment": "Portfolio Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
    elif positions_total < MAX_OPEN_TRADES and signal[-1] < 0.5:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price_bid,
            "sl": price_bid + 350 * symbol_info.point,
            "tp": price_bid - 800 * symbol_info.point,
            "deviation": 20,
            "magic": 123456,
            "comment": "Portfolio Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
    else:
        log("No signal to open a position")
        mt5.shutdown()
        return None

    for _ in range(attempts):
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # Помечаем, что по этому символу позиция открыта
            SYMBOL_TRADES[symbol] = True
            log(f"Позиция открыта {symbol}: {'Buy' if signal[-1] > 0.5 else 'Sell'}, лот={volume}, спред={spread_pips:.1f} пипсов")
            mt5.shutdown()
            return result.order
        log(f"Error: Trade request not executed, retcode={result.retcode}. Attempt {_ + 1}/{attempts}")
        time.sleep(3)
    
    mt5.shutdown()
    
def process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    log("Starting process_data function")
    augmented_data = augment_data(raw_data)
    log("Calling markup_data function")
    marked_data = markup_data(augmented_data, 'close', 'label')
    labeled_data = label_data(marked_data, SYMBOL)
    labeled_data_clustered = cluster_features_by_gmm(labeled_data, n_components=30)
    labeled_data_engineered = feature_engineering(labeled_data_clustered, n_features_to_select=11)
    return labeled_data_engineered

def evaluate_xgboost_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    y_pred = model.predict(X_test.values)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

all_symbols_done: bool = False

def process_symbol(symbol: str) -> None:
    global all_symbols_done
    try:
        raw_data = retrieve_data(symbol)
        if raw_data is None:
            log(f"Data not found for symbol {symbol}")
            return

        labeled_data_engineered = process_data(raw_data)
        train_data = labeled_data_engineered[labeled_data_engineered.index <= FORWARD]
        test_data = labeled_data_engineered[labeled_data_engineered.index > FORWARD]

        if train_data.empty or len(train_data['labels'].unique()) < 2:
            log(f"Skipping symbol {symbol}: Insufficient data or single class in labels")
            return

        xgb_clf = train_xgboost_classifier(train_data, num_boost_rounds=1000)
        test_features = test_data.drop(['label', 'labels'], axis=1)
        test_labels = test_data['labels']
        accuracy = evaluate_xgboost_classifier(xgb_clf, test_features, test_labels)
        log(f"Accuracy for symbol {symbol}: {accuracy * 100:.2f}%")

        features = test_features.values
        position_id = None
        while not all_symbols_done:
            position_id = online_trading(symbol, features, xgb_clf)
            time.sleep(6)

        all_symbols_done = True
    except Exception as e:
        log(f"Error processing symbol {symbol}: {e}")

if __name__ == "__main__":
    symbols = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURGBP"]
    
    calculate_portfolio_position_sizes(symbols)
    
    threads = []
    for symbol in symbols:
        thread = threading.Thread(target=process_symbol, args=(symbol,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

