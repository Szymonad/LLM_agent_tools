## Cyfrowe przetwarzanie i kompresja danych

Wykład 6

Operacje na obrazach. Filtracja cyfrowa.

## Dodawanie

## Operacje na obrazach

- dodawanie
- odejmowanie
- ■mnożenie i dzielenie
- ■mieszanie liniowe
- ■mieszanie metodą wagową

## Odejmowanie

- metoda: sumowanie odpowiednich elementów
- uzyskany obraz musi być poddany normalizacji (skalowaniu wartości pikseli tak, by odpowiadały poziomom występującym w obrazach oryginalnych).
- redukcja szumu – stosunek sygnał-szum w sumowanym sygnale rośnie z pierwiastkiem kwadratowym liczby sumowanych obrazów.
- SNR= N-liczba obrazów poddanych sumowaniu
- problem rozmiarów,
- problem wielkości liczby uzyskanej w wyniku sumowania - niebezpieczeństwo przekroczenia zakresu.

## Mnożenie i dzielenie obrazu (1)

- problemy skalowania i zakresu jak przy sumowaniu, zastosowanie:
- od obrazu uzyskanego z CCD odejmuje się ciemną ramkę, odpowiadającą poziomowi sygnału szumu termicznego kamery - usuwanie stalego błędu,
- nieostre maskowanie - obraz sztucznie rozmyty odejmuje się od obrazu oryginalnego w celu eliminacji składowych niskopasmowych i podniesienia kontrastu obrazu.

## Mnożenie i dzielenie obrazu (2)

- mnożeniu lub dzieleniu poddaje się wartości poszczególnych pikseli. Mnożenie (i normalizację) wykorzystuje się do uwypuklenia elementów słabo różniących się, szczególnie gdy obraz jest mało dynamiczny (zwiększenie różnicy między poziomami barw, czyli kontrastu)
- po dokonaniu mnożenia można określić poziom wartości danych, poniżej którego dane wyzerujemy – eliminacja szumu.
- dzielenie obrazu pozwala na eliminacje danych o małych wartościach (ze względu na zbliżenie wyniku do zera i zaokraglanie do wartości całkowitych). Daje to eliminację szumu.
- Problem: gdy poziom jasności obrazów dzielonych jest mały, należy przed podzieleniem przeskalować je na wyższy poziom jasności, inaczej znaczne ilości danych wpadną" do przedziału (0,1&gt; i zostaną wyzerowane. Algorytm dzielenia musi także omijać piksele o wartości 0.

```
np.: 123 125 130 131 125 14 5 do kwadratu daje 15129 15625 16900 17161 12625 196 25 dzielimy przez np. 125 121 125 135 137 125 1.5->2 0.2->0
```

---

## Mieszanie liniowe obrazów

(tzw. mieszanie alfa)

- proces odbywa się w N krokach. Polega na iteracyjnym wyznaczaniu wartości pikseli nowego obrazu ma drodze obliczeń:

jasność\_piksela\_iteracji\_n=alfa*piksel\_obrazu\_A + (1-alfa)*piksel\_obrazu\_B

- współczynnik alfa zmienia się liniowo od 0 do 1, z postępem 1/N

## Mieszanie obrazów metodą wagową (2)

- zastosowanie:
- potrzebne będą:
- a) ramka bazowa r- obraz uzyskany przy zamkniętej migawce, zawierający wyłącznie szum przyrządu
- b) ramka równomierności r - obraz uzyskany przez ekspozycję równomiernie oświetlonego tła, zawierajacy sygnał znacznie przewyższający szum. Pozwala to określić nierównomierność czułości przyrządu.
- c) r - ramka obrazu właściwego.
- od obrazu właściwego odejmuje się ramkę bazową
- rbs = r0 - - ramka obrazu bez szumu
- rrbs = r, − rbs − ramka równomiernosci bez szumu,
- rwyn = r0 / rrbs − ramka wynikowa

## Filtracja (1)

Filtrację obrazów cyfrowych uzyskuje się wykorzystując operację splotu. Operacja splotu oblicza nową wartość piksela obrazu na podstawie wartości pikseli sąsiadujących. Każdy wartość pikselu sąsiadującego jest odpowiednio ważona i wpływa na końcową wartość piksela obrazu po filtracji zgodnie ze wzorem :

Filtracja obrazów dostarcza szereg możliwości wydobycia z obrazu oryginalnego szeregu informacji lub ulatwia jego obróbkę.

## Mieszanie obrazów metodą wagową (1)

- polega na przypisaniu odpowiednich współczynników wagowych elementom poszczególnych obrazów, tak by miały mniejszy lub większy udział w obrazie wynikowym
- w odróżnieniu od mieszania liniowego dokonuje się jednokrotnie
- piksele obrazu wyjściowego mają wartości:

jasność\_piksela=waga1*piksel\_obrazu\_a +

waga2*piksel\_obrazu2

- możliwe jest mieszanie większej liczby obrazów,
- suma poszczególnych wag wynosi 1

## Filtracja cyfrowa

- filtracja dolnoprzepustowa
- ■filtracja górnoprzepustowa
- ■filtry wykrywania krawędzi

## Filtracja (2)

111

W odniesieniu do obrazu cyfrowego pojęcie pojęcie filtru dyskretnego oznacza macierz NxN np.:

## gdzie:

<!-- formula-not-decoded -->

- P,- wartość kolejnego piksela po filtracji,
- K- rząd macierzy filtru, (np. 3 -&gt; filtr 3x3),
- Pk- kolejna wartość piksela obrazu oryginalnego,
- Fk- kolejna wartość wagi filtru,
- N- norma filtru: suma wartości wag filtru, lub 1 gdy suma wynosi 0.

111

111

Operacja splotu sygnału z równaniem filtru realizowana jest na drodze mnożenia macierzy danych sygnału przez macierz filtru:

x x x

maz itu x

= nowy\_piksel

x piksel x

x x x

mnożenie typu zastapienie piksela sumą iloczynów

element przez element"

---

## Filtracja (3)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Norma to suma wartości elementów filtru lub wartość 1 gdy suma=0.

F1 to filtr uśredniajacy

Filtr jednostkowy (nie robiący nic) to 0 0 0

010

000

## Filtracja (5)

- Filtry detekcji krawędzi:

Filtr Laplacea

Nazwa tej grupy filtrów wywodzi się od równania Laplace'a, według którego zbudowano tego rodzaju filtry. Operator funkcji dwuwymiarowej zwany dal       ych drugiego rzędu danej funkcji względem każdego wymiaru co można zapisać wzorem:

dzie:

L - laplasjan,

<!-- formula-not-decoded -->

## Filtracja (7)

Pięciopunktowa maska filtru Laplace'a

Przy użyciu innych wzorów na aproksymacje drugiej pochodnej (np. stosując zależność f"(x) = [f(x)])

0 0 1 0 0

0 1 2 1 0

1 2 -16 2 1

0 1 2 1 0

0 0 1 0 0

Trzypunktowa maska filtru Laplace'a

1 -8 1

11 1

111

<!-- formula-not-decoded -->

<!-- image -->

## Filtracja (6)

Aproksymując pochodne czastkowe do postaci:

<!-- formula-not-decoded -->

operator Laplace'a możemy zapisać jako:

<!-- formula-not-decoded -->

co można wyrazić poprzez następującą maskę filtru:

0 1 0

1-4 1

0 1 0

## Filtracja (8)

- Wykrywanie krawędzi metodą przesuwania i odejmowania

W algorytmie tym odejmuje się przestrzennie przesuniętą kopię obrazu od oryginalnego obrazu. Obraz przesuwamy o jeden piksel w kierunkach poziomym, pionowym albo w obu i odejmujemy przesuniętą kopię od oryginału.

- Efekt płaskorzeźby

Efekt ten uzyskuje się dokonując splotu sygnału z równaniem filtrów do uzyskania efektu płaskorzeźby:

|   pionowy | poziomy   |
|-----------|-----------|
|      -101 | -1 -1 -1  |
|      -111 | 01 0      |

-101

11 1