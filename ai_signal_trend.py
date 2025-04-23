import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt
from xgboost import XGBClassifier
from datetime import datetime
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import optuna
import time
import MetaTrader5 as mt5
from tqdm import tqdm
from colorama import init, Fore, Style

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Symbol i konfiguracja
SYMBOL = "BTCUSD"
LOT_SIZE = 0.01
TIMEFRAME = "1h"
MAX_POSITIONS = 10
CONFIDENCE_THRESHOLD = 0.75  # Próg pewności predykcji

# Domyślne dane konta
# z jakiegoś powodu w nowej wersji MT5 jeśli wcześniej nie logowałeś się w aplikacji MT5 skypt nie przekazuje loginu i hasło i trzeba ręczenie w kreatorze aplikacji MT5
# po odpaleniu skryptu wybrać "Połącz się z istniejącym rachunkiem" i podać login i hasło, pomimo że poniżej jest wpisane
account = 12345678  # Użyj swojego numeru konta MT5
password = "twoje hasło" # Użyj swojego hasła MT5
server = "ICMarketsEU-Demo"

# Initialize Binance exchange
exchange = ccxt.binance({'rateLimit': 1200, 'enableRateLimit': True})

# Initialize colorama
init()

def run_without_mt5():
    """Uruchamia strategię bez połączenia z MT5, tylko z predykcjami sygnałów."""
    print(f"{Fore.CYAN}Uruchamiam strategię bez połączenia z MT5...{Style.RESET_ALL}")

    sl_method = get_sl_method()
    print(f"{Fore.GREEN}Wybrano metodę SL/TP: {sl_method}{Style.RESET_ALL}")

    df = fetch_data("BTC/USDT", TIMEFRAME)
    df = apply_indicators(df)
    X, y, scaler = prepare_data_for_model(df)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_model(trial, X, y), n_trials=20)

    best_params = study.best_params
    print(f"{Fore.GREEN}Najlepsze parametry: {best_params}{Style.RESET_ALL}")

    model = XGBClassifier(**best_params, objective='binary:logistic', random_state=42)
    model.fit(X, y)

    print(f"{Fore.CYAN}Uruchamiam strategię handlową (tylko predykcje)...{Style.RESET_ALL}")
    trading_strategy(model, scaler, horizon=1, sl_method=sl_method, use_mt5=False)

def test_mt5_connection(account, password, server):
    print(f"{Fore.CYAN}Testuję połączenie z MT5...{Style.RESET_ALL}")
    if not mt5.initialize():
        print(f"{Fore.RED}Błąd: Inicjalizacja MT5 nie powiodła się.{Style.RESET_ALL}")
        quit()
    if not mt5.login(account, password=password, server=server):
        print(f"{Fore.RED}Błąd: Logowanie do MT5 nie powiodło się. Kod błędu: {mt5.last_error()}{Style.RESET_ALL}")
        quit()
    else:
        print(f"{Fore.GREEN}Logowanie do MT5 udane. Konto: {account}{Style.RESET_ALL}")

def fetch_data(symbol, timeframe, since=None, limit=1000):
    print(f"{Fore.CYAN}Pobieram dane dla {symbol} w interwale {timeframe}...{Style.RESET_ALL}")
    ohlcv = []
    if since is None:
        since = exchange.parse8601("2021-01-01T00:00:00Z")
    since_dt = pd.to_datetime(since, unit='ms')
    now_dt = pd.to_datetime(datetime.utcnow())
    timeframe_seconds = pd.Timedelta(timeframe).total_seconds()
    total_candles = int((now_dt - since_dt).total_seconds() / timeframe_seconds)
    with tqdm(total=total_candles, desc="Pobieranie danych", unit="świec") as pbar:
        while True:
            try:
                data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not data:
                    break
                ohlcv.extend(data)
                since = data[-1][0] + 1
                pbar.update(len(data))
                time.sleep(exchange.rateLimit / 1000)
                if len(data) < limit:
                    break
            except Exception as e:
                print(f"{Fore.RED}Błąd pobierania danych: {e}{Style.RESET_ALL}")
                break
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"{Fore.GREEN}\nPobieranie zakończone: {len(df)} wierszy.{Style.RESET_ALL}")
    return df

def apply_indicators(df):
    print(f"{Fore.CYAN}Dodaję wskaźniki...{Style.RESET_ALL}")
    df = df.copy()
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df[['MACD', 'MACD_signal']] = macd[['MACD_12_26_9', 'MACDs_12_26_9']]
    bb = ta.bbands(df['close'], length=20, std=2)
    df[['BB_lower', 'BB_middle', 'BB_upper']] = bb[['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']]
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['EMA_200'] = ta.ema(df['close'], length=200)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['OBV'] = ta.obv(df['close'], df['volume'])
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    # Nowe cechy
    df['Momentum'] = df['close'].diff(4)  # Momentum 4-godzinne
    df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(20).mean()  # Stosunek wolumenu
    df.dropna(inplace=True)
    print(f"{Fore.GREEN}Wskaźniki dodane.{Style.RESET_ALL}")
    return df

def prepare_data_for_model(df, horizon=1):
    print(f"{Fore.CYAN}Przygotowuję dane do modelu...{Style.RESET_ALL}")
    df = df.copy()
    feature_columns = df.columns.difference(['open', 'high', 'low', 'close', 'volume'])
    X = df[feature_columns].copy()
    for col in feature_columns:
        X[f'{col}_lag1'] = X[col].shift(1)
        X[f'{col}_lag2'] = X[col].shift(2)
    X.dropna(inplace=True)
    y = np.where(df['close'].shift(-horizon) > df['close'], 1, 0)  # 1 = wzrost, 0 = spadek
    X = X.iloc[:-horizon]
    y = y[:-horizon]
    min_length = min(X.shape[0], len(y))
    X = X.iloc[:min_length]
    y = y[:min_length]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"{Fore.GREEN}Przygotowano dane: {X.shape[0]} próbek, {X.shape[1]} cech.{Style.RESET_ALL}")
    return X_scaled, y, scaler

def optimize_model(trial, X, y):
    params = {
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.8, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),  # Balans klas
        'random_state': 42
    }
    model = XGBClassifier(**params)
    tscv = TimeSeriesSplit(n_splits=5)
    accuracy_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy_scores.append(accuracy_score(y_val, y_pred))
    return np.mean(accuracy_scores)

def get_sl_method():
    valid_methods = ["atr", "percent", "support_resistance"]
    while True:
        print(f"{Fore.CYAN}\nWybierz metodę SL/TP:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}1. atr (ATR-based, multiplier=4.0, TP=4xATR){Style.RESET_ALL}")
        print(f"{Fore.WHITE}2. percent (Procentowy, max loss=0.5%, TP=1.5%){Style.RESET_ALL}")
        print(f"{Fore.WHITE}3. support_resistance (Wsparcie/Opór, lookback=20){Style.RESET_ALL}")
        choice = input(f"{Fore.CYAN}Podaj numer metody (1-3): {Style.RESET_ALL}").strip()
        if choice == "1":
            return "atr"
        elif choice == "2":
            return "percent"
        elif choice == "3":
            return "support_resistance"
        else:
            print(f"{Fore.RED}Nieprawidłowy wybór. Spróbuj ponownie.{Style.RESET_ALL}")

def calculate_sl_tp(signal, price, atr, df, sl_method):
    if sl_method == "atr":
        sl_distance = atr * 4  # Zwiększone
        tp_distance = atr * 4
        if signal == "BUY":
            sl = price - sl_distance
            tp = price + tp_distance
        else:  # SELL
            sl = price + sl_distance
            tp = price - tp_distance
    elif sl_method == "percent":
        sl_distance = price * 0.005
        tp_distance = price * 0.015
        if signal == "BUY":
            sl = price - sl_distance
            tp = price + tp_distance
        else:  # SELL
            sl = price + sl_distance
            tp = price - tp_distance
    elif sl_method == "support_resistance":
        lookback = 20
        if signal == "BUY":
            sl = df['low'].iloc[-lookback:].min()
            tp = price + 4 * atr
            sl = min(sl, price - atr * 4)
        else:  # SELL
            sl = df['high'].iloc[-lookback:].max()
            tp = price - 4 * atr
            sl = max(sl, price + atr * 4)
    else:
        raise ValueError("Nieznana metoda SL/TP")
    print(f"{Fore.YELLOW}SL: {sl:.2f}, TP: {tp:.2f}, SL Distance: {abs(price - sl):.2f}{Style.RESET_ALL}")
    return sl, tp

def send_order(signal, price, atr, df, sl_method):
    positions = mt5.positions_get(symbol=SYMBOL)
    num_positions = len(positions)

    if num_positions >= MAX_POSITIONS:
        print(f"{Fore.RED}Limit {MAX_POSITIONS} zleceń osiągnięty.{Style.RESET_ALL}")
        return None

    filling_mode = mt5.ORDER_FILLING_IOC
    sl, tp = calculate_sl_tp(signal, price, atr, df, sl_method)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
        "sl": sl,
        "tp": tp,
        "comment": f"XGBoost {signal}",
        "magic": 123456,
        "type_filling": filling_mode
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"{Fore.GREEN}Zlecenie {signal}: Sukces (Ticket: {result.order}){Style.RESET_ALL}")
        return result.order
    else:
        print(f"{Fore.RED}Błąd wysyłania zlecenia {signal}: {result.comment}, Kod: {result.retcode}{Style.RESET_ALL}")
        return None

def trading_strategy(model, scaler, horizon=1, sl_method="atr", use_mt5=True):
    last_timestamp = None
    df_history = pd.DataFrame()
    last_price = None
    open_positions = {}  # Śledzenie otwartych pozycji

    while True:
        df_new = fetch_data("BTC/USDT", TIMEFRAME, limit=200)
        if df_new.empty:
            time.sleep(60)
            continue

        df_history = pd.concat([df_history, df_new]).drop_duplicates()
        last_timestamp = df_new.index[-1]
        df_history = df_history.tail(500)

        df = apply_indicators(df_history)
        X_scaled, _, _ = prepare_data_for_model(df, horizon)
        if len(X_scaled) < 1:
            print(f"{Fore.RED}Za mało danych do predykcji.{Style.RESET_ALL}")
            time.sleep(60)
            continue

        last_row = X_scaled[-1].reshape(1, -1)
        predicted_direction = model.predict(last_row)[0]  # 1 = wzrost, 0 = spadek
        confidence = model.predict_proba(last_row)[0][predicted_direction]
        current_price = df['close'].iloc[-1]
        atr = df['ATR'].iloc[-1]

        # Filtr zmienności
        atr_mean = df['ATR'].rolling(20).mean().iloc[-1]
        if atr < atr_mean * 0.75:
            print(f"{Fore.YELLOW}Niska zmienność (ATR: {atr:.2f}), pomijam handel.{Style.RESET_ALL}")
            time.sleep(3600)
            continue

        # Filtr Bollinger Bands
        bb_width = df['BB_width'].iloc[-1]
        if bb_width < 0.03:
            print(f"{Fore.YELLOW}Wąskie Bollinger Bands ({bb_width:.4f}), pomijam handel.{Style.RESET_ALL}")
            time.sleep(3600)
            continue

        # Filtr pewności
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"{Fore.YELLOW}Niska pewność predykcji ({confidence:.2f}), pomijam handel.{Style.RESET_ALL}")
            time.sleep(3600)
            continue

        # Filtr trendu (zmodyfikowany)
        ema_diff = df['EMA_50'].iloc[-1] - df['EMA_200'].iloc[-1]
        if predicted_direction == 1 and ema_diff > atr:  # Wzrost tylko przy wyraźnym trendzie
            signal = "BUY"
            color = Fore.GREEN
        elif predicted_direction == 0 and ema_diff < -atr:  # Spadek tylko przy wyraźnym trendzie
            signal = "SELL"
            color = Fore.RED
        else:
            signal = "HOLD"
            color = Fore.YELLOW
            print(f"{color}Cena: {current_price}, Kierunek: {predicted_direction}, Pewność: {confidence:.2f}, Sygnał: {signal}{Style.RESET_ALL}")
            time.sleep(3600)
            continue

        print(f"{color}Cena: {current_price}, Kierunek: {predicted_direction}, Pewność: {confidence:.2f}, Sygnał: {signal}{Style.RESET_ALL}")

        if use_mt5:
            ticket = send_order(signal, current_price, atr, df, sl_method)
            # Śledzenie pozycji
            if ticket:
                open_positions[ticket] = {
                    'signal': signal,
                    'open_price': current_price,
                    'sl': calculate_sl_tp(signal, current_price, atr, df, sl_method)[0],
                    'tp': calculate_sl_tp(signal, current_price, atr, df, sl_method)[1]
                }

            # Sprawdzanie pozycji
            if open_positions:
                positions = mt5.positions_get(symbol=SYMBOL)
                closed_tickets = []
                for ticket, pos in open_positions.items():
                    if not any(p.ticket == ticket for p in positions):
                        # Pozycja zamknięta
                        history = mt5.history_deals_get(ticket=ticket)
                        if history:
                            close_price = history[-1].price
                            profit = history[-1].profit
                            reason = "TP" if abs(close_price - pos['tp']) < abs(close_price - pos['sl']) else "SL"
                            with open("trade_log.txt", "a") as f:
                                f.write(f"Closed: {last_timestamp}, Ticket: {ticket}, {pos['signal']}, Open: {pos['open_price']}, Close: {close_price}, Profit: {profit:.2f}, Reason: {reason}\n")
                        closed_tickets.append(ticket)
                for ticket in closed_tickets:
                    del open_positions[ticket]
        else:
            # W trybie bez MT5, pokazujemy przewidywane SL/TP
            sl, tp = calculate_sl_tp(signal, current_price, atr, df, sl_method)

        # Logowanie predykcji
        if last_price is not None:
            actual_change = (current_price - last_price) / last_price * 100
            actual_direction = 1 if current_price > last_price else 0
            with open("trade_log.txt", "a") as f:
                f.write(f"{last_timestamp}, {current_price}, Pred: {predicted_direction}, Pewność: {confidence:.2f}, {signal}, Actual: {last_price} -> {current_price} ({actual_change:.2f}%, Dir: {actual_direction})\n")
        last_price = current_price

        time.sleep(3600)

def main():
    print(f"{Fore.CYAN}Podaj dane logowania do MT5 (naciśnij Enter, aby użyć domyślnych):{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Domyślny login: {account}{Style.RESET_ALL}")
    custom_account = input(f"{Fore.CYAN}Login (lub Enter dla domyślnego): {Style.RESET_ALL}").strip()
    print(f"{Fore.WHITE}Domyślny serwer: {server}{Style.RESET_ALL}")
    custom_server = input(f"{Fore.CYAN}Serwer (lub Enter dla domyślnego): {Style.RESET_ALL}").strip()
    print(f"{Fore.WHITE}Domyślne hasło: {password}{Style.RESET_ALL}")
    custom_password = input(f"{Fore.CYAN}Hasło (lub Enter dla domyślnego): {Style.RESET_ALL}").strip()

    # Używaj domyślnych wartości, jeśli użytkownik nie podał własnych
    login = int(custom_account) if custom_account else account
    srv = custom_server if custom_server else server
    pwd = custom_password if custom_password else password

    test_mt5_connection(login, pwd, srv)

    sl_method = get_sl_method()
    print(f"{Fore.GREEN}Wybrano metodę SL/TP: {sl_method}{Style.RESET_ALL}")

    df = fetch_data("BTC/USDT", TIMEFRAME)
    df = apply_indicators(df)
    X, y, scaler = prepare_data_for_model(df)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_model(trial, X, y), n_trials=20)

    best_params = study.best_params
    print(f"{Fore.GREEN}Najlepsze parametry: {best_params}{Style.RESET_ALL}")

    model = XGBClassifier(**best_params, objective='binary:logistic', random_state=42)
    model.fit(X, y)

    print(f"{Fore.CYAN}Uruchamiam strategię handlową...{Style.RESET_ALL}")
    trading_strategy(model, scaler, horizon=1, sl_method=sl_method, use_mt5=True)

if __name__ == "__main__":
    print(f"{Fore.CYAN}Wybierz tryb uruchomienia:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}1. Z połączeniem MT5 (automatyczne zlecenia){Style.RESET_ALL}")
    print(f"{Fore.WHITE}2. Bez połączenia MT5 (tylko predykcje sygnałów){Style.RESET_ALL}")
    choice = input(f"{Fore.CYAN}Podaj numer trybu (1-2): {Style.RESET_ALL}").strip()
    if choice == "1":
        main()
    elif choice == "2":
        run_without_mt5()
    else:
        print(f"{Fore.RED}Nieprawidłowy wybór. Uruchamiam tryb z MT5.{Style.RESET_ALL}")
        main()