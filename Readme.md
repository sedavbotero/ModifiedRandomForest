**Zmodyfikowany Las Losowy (Modified Random Forest)**

Projekt ten zawiera własną implementację Drzewa Decyzyjnego oraz
Zmodyfikowanego Lasu Losowego (Modified Random Forest) do zadań
regresji.

Głównym celem projektu jest zbadanie wpływu adaptacyjnego mechanizmu
próbkowania (bootstrap) na jakość predykcji, w porównaniu do
standardowej implementacji z biblioteki scikit-learn.

**Struktura Projektu:**

-   **decisionTree.py**:

    -   Zawiera klasę DecisionTreeRegressor -- własną implementację
        drzewa regresyjnego.

    -   Obsługuje kluczowe hiperparametry: max_depth, min_samples_split,
        min_samples_leaf, min_impurity_decrease.

    -   Wykorzystuje histogramy (max_bins) do optymalizacji poszukiwania
        punktów podziału.

-   **modifiedRandomForest.py**:

    -   Zawiera klasę ModifiedRandomForest.

    -   Implementuje algorytm lasu losowego ze zmienionym mechanizmem
        doboru próby (szczegóły w sekcji Algorytm).

-   **model_testing.py**:

    -   Skrypt główny służący do uruchomienia eksperymentów.

    -   Automatycznie pobiera zbiór danych (House 8L z OpenML).

    -   Porównuje wyniki własnej implementacji z RandomForestRegressor
        ze scikit-learn.

**Opis Algorytmu (Modyfikacja)**

Standardowy Las Losowy tworzy drzewa na podstawie prób losowanych ze
zwracaniem (bootstrap) z rozkładem jednostajnym.

W **ModifiedRandomForest** prawdopodobieństwo wylosowania n-tej
obserwacji do treningu kolejnego drzewa nie jest stałe, lecz zależy od
błędu predykcji dotychczasowego zespołu drzew. Algorytm działa w sposób
przypominający boosting:

1.  Dla każdego kolejnego drzewa obliczane są wagi obserwacji na
    podstawie residuów (błędu bezwzględnego):

$$w_{i} = {|y_{i} - y_{i,akt}|}^{ex}$$

> Gdzie:

$y_{i}$: wartość rzeczywista

$y_{i,akt}$: aktualna predykcja zespołu (średnia z dotychczasowych
drzew).

$ex$: wykładnik potęgi.

2.  Wykładnik jest adaptacyjny. Zaczyna od wartości i rośnie w każdej
    iteracji, aby coraz mocniej \"karać\" trudne przypadki:

$${ex}_{t + 1} = {ex}_{t}*{1,01}^{\gamma}$$

> Aż do osiągnięcia wartości granicznej max_exponent.

Parametry sterujące tym procesem to:

-   gamma: Steruje tempem wzrostu wykładnika (szybkością adaptacji).

-   max_exponent: Maksymalna wartość wykładnika.

**Wymagania**

Projekt wymaga środowiska Python oraz następujących bibliotek:

numpy

pandas

scikit-learn

scipy

Można je zainstalować poleceniem:

pip install numpy pandas scikit-learn scipy

**Uruchomienie**

Aby uruchomić testy i porównanie modeli, należy wykonać plik
model_testing.py

**Co się wydarzy po uruchomieniu?**

1.  Skrypt sprawdzi, czy istnieje plik dataset_2204_house_8L.arff.\
    Jeśli nie, pobierze go automatycznie z OpenML.

2.  Dane zostaną wczytane i podzielone na zbiór treningowy i testowy.

3.  Nastąpi trenowanie modelu ModifiedRandomForest oraz referencyjnego
    RandomForestRegressor (sklearn).

4.  Na konsolę zostaną wypisane wyniki (MSE oraz R^2^) dla zbioru
    treningowego i testowego.
