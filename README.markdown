# AI Signal Trend 
Projekt ten powstał w 99% przy użyciu Grok AI i nie odmieni waszego życia, ma na celu zainteresowanie Was możliwościami jakie daje SI do tworzenia czegoś więcej niż śmieszne obrazki. \
Chcesz dodać integracje z binance czy twitterem ? Wklej kod do bota AI i poproś o dodanie funkcji, lub wytłumaczenie kodu, podpowiedzi jak poprawić predykcję itp. \
Powstał dla zabawy i nie ponoszę żadnej odpowiedzialności za ewentualne poniesione straty związane z jego użyciem przy spekulowaniu na giełdach za pomocą prawdziwych środków. \
Traktuj go wyłacznie w celach edukcyjnych, rozwijaj dowolnie, baw się i ucz. \
Jeśli sprawi, że zainsteresujesz się tematem to będę zadowolony, a jeśli docenisz moją prace i wesprzesz mnie za pomocą blika na zrzutka.pl to docenię to, a mój syn dostanie nowe klocki LEGO :) \
Zabawy z kodem wciągają, a dzieki AI jest to naprawdę proste. \
Jeśli liczyliście na perpetum mobile, które sprawi, że zostaniecie zilionnerami :-) to z góry muszę Was rozczarować, ale być może zachęci Was do nauki tradingu lub programowania. \
Polecam tryb z kontem brokerskim, które działa z platformą MT5, pare instrumentów możecie dowolnie zmienić w kodzie, domyśłnie jest to BTC/USD. \
Nie potraficie czegoś zainstalować ? Nie pytajcie mnie, zapytajcie AI. \
Doświadczenie z Pythonem nie jest wymagane, ale przyda się znajomosć jakiś podstaw, zmiany w kodzie warto robić w jakimś środowisku programostycznym np. w Visual Studio Code, ale może to być choćby notatnik i linia poleceń. \
Jeśli sygnał jest HOLD - czekacie na kolejny, jełśi buy owteiracie pozycje LONG, jeśłi sell to SHORT, jeśłi uruchomicie tryb z MT5 zlecenia się utworzą same [trzeba w apliakcji zezwolić na autotrade klikając |> Algo Trading \
\
Życzę miłej zabawy i powodzenia oraz samych TakeProfitów :-) !! \
\
Chcesz mnie wesprzeć? Zrób to tutaj: https://zrzutka.pl/jbg3fz \
\
`ai_signal_trend.py` to skrypt Python do automatycznego handlu BTC/USD, wykorzystujący model uczenia maszynowego XGBoost oraz dane rynkowe z Binance. Skrypt generuje sygnały handlowe (BUY, SELL, HOLD) na podstawie analizy technicznej i może wykonywać zlecenia na platformie MetaTrader5 (MT5) lub działać w trybie bez MT5, wyświetlając tylko predykcje sygnałów.

## Funkcjonalności

- **Pobieranie danych**: Dane OHLCV dla BTC/USDT z Binance (interwał 1h).
- **Analiza techniczna**: Wskaźniki takie jak RSI, MACD, Bollinger Bands, EMA, ATR, OBV, Momentum, Volume Ratio.
- **Model predykcyjny**: XGBoost z optymalizacją hiperparametrów za pomocą Optuna.
- **Filtrowanie sygnałów**: Na podstawie pewności predykcji, zmienności (ATR), szerokości Bollinger Bands i trendu (EMA).
- **Tryb z MT5**:
  - Automatyczne wysyłanie zleceń BUY/SELL na konto MT5.
  - Dynamiczne ustawianie Stop Loss (SL) i Take Profit (TP) za pomocą metod: ATR, procentowej, wsparcia/oporu.
  - **Metody SL/TP**:
    - **ATR**: Używa wskaźnika Average True Range (ATR) z mnożnikiem 4. SL i TP są obliczane jako ±4*ATR od ceny otwarcia. Metoda ta dostosowuje poziomy SL/TP do zmienności rynku, zapewniając większe odległości w okresach wysokiej zmienności.
    - **Percent**: Ustawia SL na 0,5% straty od ceny otwarcia, a TP na 1,5% zysku. Jest to metoda statyczna, niezależna od zmienności, odpowiednia dla rynków o przewidywalnej dynamice.
    - **Support/Resistance**: SL bazuje na minimach (dla BUY) lub maksimach (dla SELL) z ostatnich 20 okresów, z ograniczeniem do ±4*ATR. TP jest ustawiany na ±4*ATR. Metoda ta uwzględnia kluczowe poziomy cenowe, ale jest bardziej konserwatywna w silnych trendach.
  - Ręczne wprowadzanie loginu, hasła i serwera MT5 (z domyślnymi wartościami w kodzie).
- **Tryb bez MT5**:
  - Wyświetlanie predykcji sygnałów (BUY, SELL, HOLD) z ceną, pewnością i przewidywanymi SL/TP.
  - Brak połączenia z MT5, brak zleceń.
- **Logowanie**: Zapis predykcji i wyników transakcji do pliku `trade_log.txt`.

## Wymagania

### Systemowe

- System operacyjny: Windows, Linux lub macOS.
- Python: 3.8 lub nowszy (zalecane 3.10 lub 3.11).
- Dostęp do internetu (dla Binance API i MT5).
- MetaTrader5: Terminal MT5 i konto demo (np. ICMarketsEU-Demo) dla trybu z MT5.

### Testowane środowisko

Skrypt był uruchamiany i testowany w następującym środowisku:
- **System operacyjny**: Windows 11
- **Python**: 3.10.11 (instalowany z Microsoft Store)
- **Zależności**: Instalowane za pomocą komendy `pip install -r requirements.txt`

### Biblioteki Python

- `pandas>=2.0.0`
- `pandas-ta>=0.3.14b0`
- `numpy>=1.24.0`
- `ccxt>=4.0.0`
- `xgboost>=2.0.0`
- `scikit-learn>=1.3.0`
- `optuna>=3.0.0`
- `MetaTrader5>=5.0.45` (tylko dla trybu z MT5)
- `tqdm>=4.65.0`
- `colorama>=0.4.6`

## Instalacja

1. **Zainstaluj Pythona**:

   - Pobierz i zainstaluj Pythona (https://www.python.org/downloads/) lub użyj Microsoft Store dla Windows.
   - Sprawdź wersję: `python --version` lub `python3 --version`.

2. **Zainstaluj biblioteki**:

   Użyj dołączonego pliku `requirements.txt`, który jest gotowy do pobrania:

   ```bash
   pip install -r requirements.txt
   ```

3. **Skonfiguruj MetaTrader5** (tylko dla trybu z MT5):

   - Pobierz i zainstaluj terminal MT5 od brokera (np. https://www.icmarkets.com).
   - Utwórz konto demo i zapisz dane logowania (login, hasło, serwer).
   - Dodaj symbol `BTCUSD` do okna **Market Watch** w MT5.

4. **Pobierz skrypt**:

   - Skopiuj plik `ai_signal_trend.py` oraz `requirements.txt` do swojego katalogu roboczego.

## Uruchomienie

1. **Uruchom skrypt**:

   ```bash
   python ai_signal_trend.py
   ```

2. **Wybierz tryb**:

   - Po uruchomieniu skrypt zapyta o tryb:

     ```
     Wybierz tryb uruchomienia:
     1. Z połączeniem MT5 (automatyczne zlecenia)
     2. Bez połączenia MT5 (tylko predykcje sygnałów)
     Podaj numer trybu (1-2):
     ```

### Tryb z MT5 (opcja 1)

- **Wprowadzanie danych logowania**:

  - Skrypt wyświetli domyślne dane logowania:

    ```
    Podaj dane logowania do MT5 (naciśnij Enter, aby użyć domyślnych):
    Domyślny login: 12345678
    Login (lub Enter dla domyślnego): 
    Domyślny serwer: ICMarketsEU-Demo
    Serwer (lub Enter dla domyślnego): 
    Domyślne hasło: **gK@wZw******
    Hasło (lub Enter dla domyślnego): 
    ```

  - Naciśnij Enter, aby użyć domyślnych wartości, lub wpisz własne dane logowania.

- **Wybór metody SL/TP**:

  - Wybierz jedną z metod: `atr`, `percent`, `support_resistance`.

- **Działanie**:

  - Skrypt łączy się z MT5, pobiera dane z Binance, trenuje model i wystawia zlecenia. Nie zapomnij dodać  wirtualnych środków na stronie brokera do portfela DEMO
  - Wyniki są zapisywane do `trade_log.txt`.

### Tryb bez MT5 (opcja 2)

- **Wybór metody SL/TP**:
  - Wybierz metodę, jak w trybie z MT5.
- **Działanie**:
  - Skrypt pobiera dane, generuje predykcje i wyświetla sygnały (BUY, SELL, HOLD) z ceną, pewnością i SL/TP.
  - Nie łączy się z MT5 ani nie wystawia zleceń.
  - Predykcje są zapisywane do `trade_log.txt`.

## Struktura pliku `trade_log.txt`

- **Predykcje**:

  ```
  2025-04-22 10:00:00, 60000.0, Pred: 1, Pewność: 0.85, BUY, Actual: 59000.0 -> 60000.0 (1.69%, Dir: 1)
  ```

  - Format: `timestamp, cena, przewidywany kierunek, pewność, sygnał, rzeczywista zmiana ceny`.

- **Zamknięte pozycje** (tylko tryb z MT5):

  ```
  Closed: 2025-04-22 10:00:00, Ticket: 123456, BUY, Open: 60000.0, Close: 62000.0, Profit: 20.00, Reason: TP
  ```

  - Format: `timestamp, ticket, sygnał, cena otwarcia, cena zamknięcia, zysk/strata, powód zamknięcia`.

## Uwagi

- **Bezpieczeństwo**:
  - Używaj konta demo do testowania, aby uniknąć ryzyka finansowego.
  - Domyślne dane logowania w kodzie (`account`, `password`, `server`) powinny być zmienione na własne.
- **Problemy z MT5**:
  - Upewnij się, że terminal MT5 jest uruchomiony i symbol `BTCUSD` jest dostępny w **Market Watch**.
  - Sprawdź poprawność danych logowania i połączenia internetowego.
- **Problemy z Binance**:
  - Jeśli pojawi się błąd `RateLimitExceeded`, zwiększ `rateLimit` w kodzie (np. `'rateLimit': 2000`).
  - Niektóre kraje mogą wymagać VPN do dostępu do Binance.
- **Wydajność**:
  - Pobieranie dużych ilości danych (od 2021) może wymagać kilku GB RAM.
  - Rozważ zapis danych do pliku CSV dla szybszego ponownego użycia.

## Licencja

MIT License

## Autorstwo

- **Kod źródłowy**: Skrypt `ai_signal_trend.py` wykorzystuje fragmenty kodu autorstwa FROST, opublikowanego w artykule „Advanced Bitcoin Signal Bot for Predicting Trades” na platformie Medium.com.\
  https://medium.com/coinmonks/advanced-bitcoin-signal-bot-to-predict-trades-d784a96f34e6 \
  Jest to zoptymalizowana wersja oryginalnego kodu, poparta tygodniami testów, wzbogacona o dodatkowe funkcje i wskaźniki techniczne rozszerzające jego możliwości.

---
**Donate:**  https://zrzutka.pl/jbg3fz \
**Autor**: GeorgeZip\
**Kontakt**: nie pytaj mnie, pytaj AI \
**Data**: Kwiecień 2025
