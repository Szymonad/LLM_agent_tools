## Cyfrowe przetwarzanie kompresja danych

Wykład 6

Operacje na obrazach. Filtracja cyfrowa.

## Operacje na obrazach

- dodawanie
- odejmowanie
- mnożenie i dzielenie
- mieszanie liniowe
- mieszanie metodą wagową

## Dodawanie

- metoda: sumowanie odpowiednich elementów
- uzyskany obraz musi być poddany normalizacji (skalowaniu wartości pikseli tak, by odpowiadały poziomom występującym w obrazach oryginalnych) -
- SNR= N-liczba obrazów poddanych sumowaniu
- redukcja szumu stosunek sygnał-szum w sumowanym sygnale rośnie z pierwiastkiem kwadratowym liczby sumowanych obrazów
- problem rozmiarów ,
- problem wielkości liczby uzyskanej w wyniku sumowania niebezpieczeństwo przekroczenia zakresu.

## Mnożenie i dzielenie obrazu (1)

- mnożeniu lub dzieleniu poddaje się wartości poszczególnych pikseli. Mnozenie (i normalizację) wykorzystuje się do uwypuklenia elementów słabo różniących się, szczególnie gdy obraz jest mało dynamiczny (zwiększenie różnicy między poziomami barw , czyli kontrastu)

```
123 125 130 131 125 14 5 do kwadratu daje 15129 15625 16900 17161 12625 196 25 |dzielimy przez np. 125 121 125 135 137 125
```

## Odejmowanie

- problemy skalowania zakresu jak przy sumowaniu, zastosowanie:
- od obrazu uzyskanego z CCD odejmuje się ciemną ramke, odpowiadająca poziomowi sygnału szumu termicznego kamery usuwanie stałego błędu,
- nieostre maskowanie obraz sztucznie rozmyty odejmuje się od obrazu oryginalnego w celu eliminacji składowych niskopasmowych podniesienia kontrastu obrazu.

## Mnożenie i dzielenie obrazu (2)

- po dokonaniu mnożenia można określić poziom wartości danych, poniżej którego dane wyzerujemy eliminacja szumu.
- dzielenie obrazu pozwala na eliminacje danych 0 małych wartościach (ze względu na zbliżenie wyniku do zera zaokrąglanie do wartości całkowitych) . Daje to eliminację szumu.
- Problem: gdy poziom jasności obrazów dzielonych jest mały , należy przed podzieleniem przeskalować je na wyższy poziom jasności , inaczej znaczne ilości danych ,wpadną' do przedziału (0,1&gt; zostaną wyzerowane. Algorytm dzielenia musi także omijać piksele 0 wartości 0.

---

## Mieszanie liniowe obrazów

(tzw. mieszanie alfa)

- proces odbywa się w N krokach. Polega na iteracyjnym wyznaczaniu wartości pikseli nowego obrazu ma drodze obliczeń:

|jasność\_piksela\_iteracji\_n=alfa*piksel\_obrazu\_A (1-alfa)*piksel\_obrazu\_B

- współczynnik alfa zmienia się liniowo od 0 do 1, 2 postępem 1/N

## Mieszanie obrazów metodą wagową (2)

- zastosowanie:
- potrzebne będą:
- a) ramka bazowa Fp obraz uzyskany przy zamkniętej migawce, zawierający wyłącznie szum przyrządu

## Mieszanie obrazów metodą wagową (1)

- polega na przypisaniu odpowiednich współczynników wagowych elementom poszczególnych obrazów, tak by miały mniejszy lub większy udział w obrazie wynikowym
- W odróżnieniu od mieszania liniowego dokonuje się jednokrotnie
- piksele obrazu wyjściowego mają wartości:

waga2*piksel\_obrazu2

[jasność\_piksela=wagal*piksel\_obrazu

- możliwe jest mieszanie większej liczby obrazów,
- suma poszczególnych wag wynosi 1

## Filtracja cyfrowa

- filtracja dolnoprzepustowa
- wykrywania krawędzi filtry
- b) ramka równomierności rr obraz uzyskany przez ekspozycję równomiernie oświetlonego tła, zawierający sygnał znacznie przewyższający szum. Pozwala to określić nierównomierność czułości przyrządu.
- c) ro ramka obrazu właściwego.
- od obrazu właściwego odejmuje się ramkę bazową
- Fbs To Tb ramka obrazu bez szumu
- Frbs Tr Tbs ramka równomiernosci bez szumu,
- To Frbs ramka wynikowa Twn

## Filtracja (1)

[Filtrację obrazów cyfrowych uzyskuje się wykorzystując operacje splotu. Operacja splotu oblicza nowa wartość piksela obrazu na podstawie wartości pikseli sąsiadujących. Każdy wartość pikselu sąsiadujacego jest odpowiednio wazona wpływa na końcowa wartość piksela obrazu po filtracji zgodnie ze wzorem

[Filtracja obrazów dostarcza szereg możliwości wydobycia z obrazu oryginalnego szeregu informacji lub ułatwia jego obróbkę.

- filtracja górnoprzepustowa

## Filtracja (2)

W odniesieniu do obrazu cyfrowego pojęcie pojęcie filtru dyskretnego oznacza maderz   NxN np.

## Igdzie:

- P - wartość kolejnego piksela po filtracji ,
- K - rząd macierzy filtru, (np. 3 -&gt; filtr 3x3),
- kolejna wartość piksela obrazu oryginalnego,
- Fu - kolejna wartość wagi filtru,
- N norma filtru: suma wartości wag filtru, lub 1 gdy suma wynosi 0.

<!-- formula-not-decoded -->

1 1

Operacja splotu sygnału z równaniem filtru realizowana jest na drodze mnozenia macierzy danych sygnału przez macierz filtru:

X piksel x macierz filtru nowy\_piksel

mnozenie typu zastapienie piksela sumą iloczynów

"element przez element"

---

## Filtracja (3)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(Norma to suma wartośa elementów filtru lub wartość 1 gdy suma=0.

[F1 to filtr uśredniający

[Filtr jednostkowy (nie robiący nic) to 0 0 0

0 1 0

0 0 0

## Filtracja (5)

- Filtry detekcji krawędzi:

[Filtr Laplacea

[Nazwa tej grupy filtrów wywodzi się od równania Laplace'a, według którego zbudowano tego rodzaju filtry. Operator funkcji dwuwymiarowej zwany dalej laplasjanem jest okreslany Jako suma pochodnych cząstkowych drugiego rzędu danej funkcji względem kazdego wymiaru co mozna zapisać wzorem:

laplasjan,

<!-- formula-not-decoded -->

## Filtracja (7)

Przy   uzyciu innych wzorów na aproksymacje drugiej pochodnej stosując zalezność f"(x) = [f(x)]) (np.

[Pięciopunktowa maska filtru Laplace'a

1 0

2 1

1 2-16 2

2 1

0 0 1 0

Trzypunktowa maska filtru Laplace'a

1 1 1

1-8

<!-- image -->

## Filtracja (6)

Aproksymując pochodne cząstkowe do postaci:

operator Laplace'a możemy zapisać jako:

144

## Filtracja (8)

- Wykrywanie krawędzi metodą przesuwania odejmowania

W algorytmie tym odejmuje się przestrzennie przesuniętą kopię obrazu od oryginalnego obrazu. Obraz przesuwamy 0 jeden piksel w kierunkach poziomym, pionowym albo w obu i odejmujemy przesuniętą kopie od oryginału.

- Efekt płaskorzeźby

Efekt ten uzyskuje się dokonując splotu sygnału z równaniem filtrów do uzyskania efektu płaskorzeźby:

|   pionowy | poziomy   |
|-----------|-----------|
|           | 41-1-1    |
|      4111 | 0 1       |
|           | 1 1       |