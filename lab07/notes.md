# Decision Transformer (DT)

Offline Reinforcement Learning - czyli nie mamy dostępu do środowiska (przynajmniej podczas treningu). Walidacja/ewaluacja już pewnie ma dostęp do środowiska, bo trzeba jakoś sprawdzić. Trzbea mć dataset/trajektorie kogoś innego - eksperta. Nie musi być perfekcyjny, to jest poza naszą kontrolą. Na podstawie tego zbioru danych chcemy nauczyć politykę, która podejmuje optymalne decyzje, akcje.

W zbiorze danych mamy przestrzeń stanów i akcji, które odwiedziliśmy i podjęliśmy, ale nie wiemy, co znajduje się poza tym obszarem - być może to tam znajduje się optymalna polityka. Możemy znajdować polityki, które podejmują najlepsze/optymalne decyzje w tej przestrzeni, którą widzieliśmy.

Można to porównać do uczenia nadzorowanego (supervised learning). Zbiór trajektori wygenerowany przez eksperta (agent wytrenowany przez online RL często). Jednym podejściem jest Behavioral Cloning - robi to samo, co ekspert z danych - używany go jako baseline.

Bo celem trenowania nowego agenta jest to, aby był lepszy od eksperta. Idealnym sposobem było by zrozumienie środowiska, stworzenie modelu środowiska, tak, że jestem w stanie poradzić sobie z nim lepiej niż agent, którego obserwowałem. Mogą też powtarzać się te same stany z różną akcją - można wtedy sprawdzić, która dała lepszą nagrodę i ją wybrać.

Online On Policy
Online Off Policy - to się nadaje

Wada: krytyk przeszacowuje wartości stanów, których nie widział - wyprowadza aktora w pole.

Offline RL - jak modelowanie sekwencji: STAN AKCJA NAGRODA STAN AKCJA.

Tokeny nie mogą być tylko stanami, bo będziemy przewidywać tylko kolejne stany, bez nagród, czy akcji. Wrzucamy $\hat{r}$ - przyszły zwrot, stan akcja - taka trójka (trzy tokeny). Chcemy w wyniku dostać następną akcję. Wrzucam więcej tokenów, niż pobieram.

$\hat{r}$ - r z daszkiem, kluczowy pomysł. Gdybyśmy wrzuiccili stan, akcja, nagroda - dla tego stanu, zwróć akcję, ale nie bardzo jak to wykorzystać podczas ewaluacji. Podczas ewaluacji chcę zmusić model do zwracania optymalnej akcji, ale żeby wiedział, które są optymalne to muszę wprowadzić tę informację. Pomysłem jest wprowadzenie, ze r z daszkiem to suma przyszłych nagród, którą znam bo to jest offline-dataset (mogę policzyć), często nie liczę do końca epizodu, ale okno. Trenujemy politykę $\pi(a|s, r z daszkiem)$. Podczas ewaluacji będziemy już podawać możliwą wysoką wartość nagrody skumulowanej.