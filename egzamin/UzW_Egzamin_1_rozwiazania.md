# Uczenie ze Wzmocnieniem - Egzamin 1 - ROZWIĄZANIA

> Rozwiązania do pliku `UzW_Egzamin_1_zadania.md`.
> Zadania 4 i 6 są otwarte: podano wzorcowy szkielet odpowiedzi, a nie jedyne dobre rozwiązanie.

---

## Zadanie 1 - Aktor-Krytyk, jeden krok uczący

### Dane i oznaczenia

$s_1=[x,y]=[3,1]$, wykonana akcja $a_1$, $s_2=[1,1]$, $r=10$, $\gamma=1$, $\alpha=0.5$.
Parametry startowe: $\theta_a=2,\ \theta_b=2,\ w=2$.

### Krok 1: rozkład polityki w stanie $s_1$ (stan wyjściowy)

$$\theta_a x = 2\cdot 3 = 6,\qquad \theta_b y = 2\cdot 1 = 2,\qquad \text{suma} = 8.$$

$$\pi(a_0\mid s_1)=\frac{6}{8}=0.75,\qquad \pi(a_1\mid s_1)=\frac{2}{8}=0.25.$$

Agent wykonał akcję mniej prawdopodobną ($a_1$, p-stwo 0.25). To efekt eksploracji.

### Krok 2: błąd TD (pełni rolę advantage)

$$v_w(s_1)=w(x+y)=2\cdot(3+1)=8,\qquad v_w(s_2)=2\cdot(1+1)=4.$$

$$\delta = r + \gamma\,v_w(s_2) - v_w(s_1) = 10 + 4 - 8 = 6.$$

> Uwaga: $\delta>0$ oznacza, że wynik był lepszy niż przewidywał krytyk. Skoro wykonano akcję $a_1$, krok powinien **zwiększyć** preferencję dla $a_1$.

### Krok 3: aktualizacja krytyka

Półgradientowa reguła SGD: $w \leftarrow w + \alpha\,\delta\,\nabla_w v_w(s_1)$, gdzie $\nabla_w v_w(s_1)=x+y=4$.

$$w \leftarrow 2 + 0.5\cdot 6\cdot 4 = 2 + 12 = \boxed{14}.$$

Sprawdzenie sensowności: celem aktualizacji była wartość $r+v_w(s_2)=10+4=14$, a stara estymata $v_w(s_1)=8$. Po aktualizacji nowa estymata to $v_{14}(s_1)=14\cdot4=56$, czyli **mocno przestrzeliliśmy** cel (14). Wniosek w komentarzu poniżej.

### Krok 4: aktualizacja aktora

Reguła: $\theta \leftarrow \theta + \alpha\,\delta\,\nabla_\theta \ln\pi(a_1\mid s_1)$.

Dla $\pi(a_1\mid s)=\dfrac{\theta_b y}{\theta_a x+\theta_b y}$ mamy $\ln\pi(a_1)=\ln(\theta_b)+\ln(y)-\ln(\theta_a x+\theta_b y)$, więc:

$$\frac{\partial \ln\pi(a_1)}{\partial \theta_a} = -\frac{x}{\theta_a x+\theta_b y} = -\frac{3}{8} = -0.375,$$

$$\frac{\partial \ln\pi(a_1)}{\partial \theta_b} = \frac{1}{\theta_b} - \frac{y}{\theta_a x+\theta_b y} = \frac12 - \frac18 = 0.375.$$

Mnożnik $\alpha\delta = 0.5\cdot 6 = 3$:

$$\theta_a \leftarrow 2 + 3\cdot(-0.375) = 2 - 1.125 = \boxed{0.875},$$
$$\theta_b \leftarrow 2 + 3\cdot(0.375) = 2 + 1.125 = \boxed{3.125}.$$

### Krok 5: wpływ na późniejsze preferencje

Nowy rozkład polityki w stanie $s_1$:

$$\theta_a x = 0.875\cdot 3 = 2.625,\qquad \theta_b y = 3.125\cdot 1 = 3.125,\qquad \text{suma}=5.75.$$

$$\pi(a_1\mid s_1)=\frac{3.125}{5.75}\approx 0.543,\qquad \pi(a_0\mid s_1)\approx 0.457.$$

**Interpretacja:** prawdopodobieństwo akcji $a_1$ wzrosło z $0.25$ do ok. $0.54$. To poprawny kierunek, bo advantage był dodatni, a wykonano właśnie $a_1$. Aktor preferuje teraz $a_1$ także w innych stanach, w których $\theta_b y$ dominuje nad $\theta_a x$ (czyli tam, gdzie $y$ jest względnie duże).

**Komentarz o $\alpha$:** krok $\alpha=0.5$ przy module cechy krytyka równym 4 daje efektywny skok wartości $0.5\cdot 4^2=8$ na jednostkę $\delta$, czyli krytyk przeskoczył cel 14 aż do 56. To sygnał, że dla tej aproksymacji $\alpha=0.5$ jest **za duże** i grozi rozbieżnością. Wykład wprost podkreśla: przy bootstrappingu i aproksymacji trzeba robić małe kroki. Bezpieczniej byłoby np. $\alpha\approx 0.05$ lub normalizacja cech.

---

## Zadanie 2 - wielorękie bandyty

### (a) zachłanna i $\varepsilon$-zachłanna

Wybór zachłanny: $\arg\max Q = a_2$ (bo $Q(a_2)=1.0$ jest największe).

W $\varepsilon$-zachłannej z $\varepsilon=0.3$ przez 70% kroków wybierane jest $a_2$ (eksploatacja), a w 30% kroków losujemy spośród wszystkich 3 akcji jednostajnie. Ułamek wszystkich wyborów trafiających w $a_2$:

$$P(a_2) = (1-\varepsilon) + \varepsilon\cdot\frac{1}{3} = 0.7 + 0.3\cdot\frac13 = 0.7 + 0.1 = \boxed{0.8}.$$

(Pozostałe akcje: po $0.3\cdot\frac13=0.1$ każda.)

### (b) UCB1

$\ln t = \ln 10 \approx 2.3026$. Liczymy $Q(a)+\sqrt{\ln t / N(a)}$:

| akcja | $Q$ | $N$ | bonus $\sqrt{\ln t / N}$ | suma |
|:-----:|:---:|:---:|:------------------------:|:----:|
| $a_1$ | 0.8 | 1 | $\sqrt{2.3026/1}\approx 1.517$ | $\approx 2.317$ |
| $a_2$ | 1.0 | 8 | $\sqrt{2.3026/8}\approx 0.536$ | $\approx 1.536$ |
| $a_3$ | 0.5 | 1 | $\sqrt{2.3026/1}\approx 1.517$ | $\approx 2.017$ |

Wybór UCB1: $\boxed{a_1}$ (suma 2.317).

**Dlaczego inaczej niż zachłannie:** akcja $a_2$ ma najwyższą estymatę, ale była wybierana aż 8 razy, więc jej składnik niepewności jest mały. Akcje $a_1$ i $a_3$ były próbowane tylko raz, więc dostają duży bonus eksploracyjny. UCB premiuje akcje słabo zbadane, by zmniejszyć ryzyko przeoczenia lepszej opcji.

### (c) gradientowy bandyta (soft-max)

Start $H=[0,0,0]$, więc $\pi=[\tfrac13,\tfrac13,\tfrac13]$. Wybrano $A_t=a_2$, $R=1$, $\bar R=0.5$, $\alpha=0.5$. Czynnik $\alpha(R-\bar R)=0.5\cdot 0.5=0.25$.

Akcja wybrana ($a_2$): $\mathbb{1}-\pi = 1-\tfrac13=\tfrac23$:
$$H(a_2)\leftarrow 0 + 0.25\cdot \tfrac23 = \tfrac{1}{6}\approx 0.1667.$$

Akcje niewybrane ($a_1,a_3$): $\mathbb{1}-\pi = 0-\tfrac13=-\tfrac13$:
$$H(a_1)=H(a_3)\leftarrow 0 + 0.25\cdot(-\tfrac13) = -\tfrac{1}{12}\approx -0.0833.$$

Nowe preferencje: $H\approx[-0.0833,\ 0.1667,\ -0.0833]$.

Nowy rozkład soft-max ($e^{0.1667}\approx1.1814$, $e^{-0.0833}\approx0.9201$, suma $\approx 3.0216$):

$$\pi(a_2)\approx \frac{1.1814}{3.0216}\approx 0.391,\qquad \pi(a_1)=\pi(a_3)\approx \frac{0.9201}{3.0216}\approx 0.305.$$

Prawdopodobieństwo $a_2$ wzrosło z $0.333$ do ok. $0.391$, bo nagroda przewyższyła średnią.

### (d) Thompson / rozkład Beta

Sprzężenie Beta-Bernoulli: a priori $\mathrm{Beta}(\alpha,\beta)=\mathrm{Beta}(1,1)$ (jednostajny), 3 sukcesy i 2 porażki dają a posteriori:

$$\mathrm{Beta}(1+3,\ 1+2) = \boxed{\mathrm{Beta}(4,\,3)}.$$

Wartość oczekiwana: $\dfrac{\alpha}{\alpha+\beta}=\dfrac{4}{7}\approx 0.571$ (moda: $\tfrac{4-1}{4+3-2}=\tfrac35=0.6$).

**Jak używa tego Thompson:** w każdym kroku dla **każdej** akcji losujemy próbkę z jej rozkładu a posteriori (tu $\theta\sim\mathrm{Beta}(4,3)$), a następnie wybieramy akcję o najwyższej wylosowanej wartości. Akcje mało zbadane mają szerokie rozkłady, więc czasem wylosują wysoką wartość i zostaną zbadane. To naturalny, probabilistyczny mechanizm balansu eksploracja-eksploatacja, bez ręcznego strojenia $\varepsilon$.

---

## Zadanie 3 - teoria

### (a) równania Bellmana

Dla polityki $\pi$:

$$v_\pi(s)=\sum_a \pi(a\mid s)\sum_{s',r} p(s',r\mid s,a)\big[\,r+\gamma\,v_\pi(s')\,\big].$$

Składniki: $\pi(a\mid s)$ to prawdopodobieństwo wyboru akcji przez agenta; $p(s',r\mid s,a)$ to dynamika środowiska; $r$ to nagroda natychmiastowa; $\gamma v_\pi(s')$ to zdyskontowana wartość stanu następnego. Wartość stanu to oczekiwany zwrot uśredniony po akcjach polityki i po reakcjach środowiska.

Równanie optymalności:

$$v_*(s)=\max_a \sum_{s',r} p(s',r\mid s,a)\big[\,r+\gamma\,v_*(s')\,\big].$$

**Co znika:** suma ważona po akcjach z wagami $\pi(a\mid s)$ zostaje zastąpiona operatorem $\max_a$. Oznacza to, że pozbywamy się zależności od konkretnej polityki: optymalny agent w każdym stanie po prostu wybiera akcję o najwyższej oczekiwanej wartości, a nie miesza akcje według jakiegoś $\pi$.

### (b) próbkowanie istotności: zwyczajne vs ważone

Niech $\rho$ to iloczyn ilorazów istotności (*importance ratio*) wzdłuż trajektorii.

- **Zwyczajne (ordinary):** $V(s)=\frac{1}{n}\sum \rho\,G$. Jest **nieobciążone** (wartość oczekiwana równa $v_\pi$), ale ma **potencjalnie nieograniczoną wariancję**, bo czynniki $\rho$ mogą przyjmować bardzo duże wartości.
- **Ważone (weighted):** $V(s)=\frac{\sum \rho\,G}{\sum \rho}$. Jest **obciążone** (dla jednej próbki jego wartość oczekiwana to $v_b$, nie $v_\pi$), lecz obciążenie maleje do zera wraz z liczbą próbek. Ma **ograniczoną wariancję**, bo udział każdego składnika sumy jest nie większy niż 100%.

**Pętle w MDP:** każde okrążenie pętli dokłada kolejny czynnik do iloczynu wag (np. jeśli polityka zachowania wybiera dany ruch z p-stwem 0.5, a docelowa z p-stwem 1, czynnik to $1/0.5=2$). Iloczyn wag rośnie wykładniczo z liczbą okrążeń, co rozsadza wariancję wariantu zwyczajnego. Wariant ważony jest tu znacznie odporniejszy.

### (c) śmiertelna trójca

Trzy składniki, które osobno są nieszkodliwe, ale razem grożą rozbieżnością:

1. **Aproksymacja funkcji** wartościującej (konieczna, by generalizować na duże przestrzenie stanów).
2. **Bootstrapping** (aktualizacja estymaty na podstawie innej estymaty, jak w TD/DP; przyspiesza uczenie i zmniejsza wariancję).
3. **Uczenie poza-polityką** (rozdzielenie polityki docelowej od zachowania).

**Przykład $w$, $2w$:** dwa stany aproksymowane jako $w$ i $2w$, przejście z pierwszego do drugiego z nagrodą 0, start $w=10$. Cel TD dla pierwszego stanu to $0+\gamma\cdot 2w = 20$ (przy $\gamma\approx1$), czyli o 10 więcej niż obecne $w=10$. Krok $\alpha=0.1$ podbija $w$ w stronę 11, co automatycznie podbija drugi stan do 22, więc błąd jeszcze rośnie, kolejna aktualizacja jest większa i estymaty uciekają do nieskończoności. W uczeniu z-polityką ratuje nas to, że drugi stan trzeba kiedyś opuścić, więc jego (zawyżone) wartościowanie zostanie zweryfikowane i spadnie. W uczeniu poza-polityką próbkowanie istotności może pomijać te korygujące przejścia, więc zabezpieczenie znika.

---

## Zadanie 4 - projekt agenta sterującego grupą wind (szkic wzorcowy)

> Nie ma jednej poprawnej odpowiedzi. Oceniana jest spójność i świadomość kompromisów.

### (a) sformułowanie i wybory projektowe

**MDP:**
- **Stan:** wektor cech opisujący sytuację: dla każdego piętra flagi wezwań góra/dół i czas ich oczekiwania; dla każdej kabiny: pozycja, kierunek, zbiór wciśniętych przycisków, obłożenie (jeśli mierzone); cechy kontekstu (pora dnia, dzień tygodnia) wychwytujące niestacjonarność ruchu.
- **Akcje:** dla każdej windy decyzja w punktach decyzyjnych (zatrzymać się na piętrze / minąć / zmienić kierunek / czekać). Wygodnie traktować to jako system wieloagentowy (jedna polityka współdzielona przez wszystkie windy) lub jeden agent centralny.
- **Nagroda:** ujemny skumulowany czas oczekiwania i przejazdu, np. $-\sum (\text{czas oczekiwania pasażerów})$ na krok, ewentualnie z karą za przepełnienie. Można dodać karę za zużycie energii.

**Algorytm:** problem ma dużą, częściowo obserwowalną przestrzeń stanów i ciągły dopływ danych, więc rozsądny jest **aktor-krytyk z aproksymacją sieciami neuronowymi** (np. A2C/PPO). PPO daje stabilność przy długim treningu. Alternatywnie deep Q-learning, jeśli akcje pozostaną dyskretne i nieliczne.

**Reprezentacja:** sieć neuronowa nad wektorem cech (ewentualnie z osobnym kodowaniem per-winda i agregacją, by obsłużyć zmienną liczbę wind).

**Eksploracja:** soft-max/entropia w polityce (naturalne dla aktor-krytyk) lub $\varepsilon$-zachłanność dla DQN.

**Trening:** w **symulatorze** ruchu pasażerskiego z realistycznymi rozkładami przyjazdów (szczyt poranny w górę, popołudniowy w dół). Wykład wprost zauważa, że często łatwiej symulować proces zgodnie z rozkładami niż podać je jawnie. Po treningu walidacja na podmienionych profilach ruchu, dopiero potem wdrożenie z możliwością douczania online.

### (b) trzy uzasadnienia

1. **Aktor-krytyk zamiast czystego wartościowania:** akcje wind można traktować dyskretnie, ale polityka stochastyczna ułatwia eksplorację i radzi sobie z częściową obserwowalnością (nie znamy celów pasażerów), gdzie optymalna może być polityka miękka.
2. **Symulacja zamiast uczenia na żywym budynku:** trening na prawdziwych windach byłby wolny, drogi i ryzykowny (frustracja użytkowników). Symulacja daje miliony epizodów i bezpieczną eksplorację.
3. **Cechy pory dnia w stanie:** ruch jest niestacjonarny; bez kontekstu czasowego jedna polityka musiałaby uśredniać sprzeczne wzorce. Z kontekstem agent może uczyć się odmiennych strategii dla szczytu w górę i w dół.

### (c) najważniejsze ryzyko i jego ograniczenie

Najpoważniejsze ryzyko: **rozbieżność symulacja-rzeczywistość** (*sim-to-real gap*). Jeśli symulator zakłada inne rozkłady przyjazdów, czasy wsiadania czy awarie niż realny budynek, świetna polityka z symulacji może działać słabo lub destabilizować się na żywo (to także scenariusz śmiertelnej trójcy z poprzedniego zadania).

Ograniczenie prawdopodobieństwa:
- **randomizacja domeny** w symulacji (losowanie natężeń, czasów, profili), by polityka była odporna na rozbieżności parametrów;
- **kalibracja symulatora** danymi z czujników rzeczywistego budynku przed wdrożeniem;
- **ostrożne wdrożenie**: najpierw tryb cienia (polityka tylko proponuje, decyduje stary kontroler), monitorowanie metryk, dopiero potem przejęcie sterowania z możliwością szybkiego rollbacku;
- **małe kroki uczące** i regularyzacja przy ewentualnym douczaniu online, by uniknąć eksplozji wartościowania.

---

## Zadanie 5 - quiz

### (a) średnia wykładniczo ważona aktualnością

Przy stałym $\alpha$ świeższe nagrody mają większą wagę, a starsze gasną geometrycznie. To **pożądane w środowiskach niestacjonarnych**, bo estymata podąża za zmieniającym się rozkładem zamiast uśredniać całą (nieaktualną) historię.

Warunki zbieżności: $\sum_n \alpha_n = \infty$ jest **spełniony** (suma stałych jest nieskończona), ale $\sum_n \alpha_n^2 < \infty$ jest **niespełniony** (suma stałych kwadratów też jest nieskończona). Estymata zatem **nie zbiega się** do jednej wartości w sensie pewnym, lecz nieustannie fluktuuje wokół bieżącej prawdy. W problemie niestacjonarnym jest to cecha pożądana, nie wada.

### (b) liczba kroków planowania w Dyna-Q

Więcej kroków planowania $n$ na jeden krok rzeczywisty oznacza, że informacja z każdego rzeczywistego doświadczenia jest wielokrotnie propagowana przez model do funkcji wartościującej. Skutek: **mniej rzeczywistych interakcji** potrzebnych do nauczenia się dobrej polityki (szybsze uczenie liczone w krokach środowiska), kosztem **większego nakładu obliczeniowego** na krok. Pomaga najbardziej, gdy rzeczywiste doświadczenie jest **drogie lub wolne do zdobycia**, a posiadany **model jest dokładny** (przy idealnym, deterministycznym modelu planowanie działa jak dodatkowe przejścia programowania dynamicznego).

### (c) Expected SARSA

Zwykła SARSA używa wartości jednej **wylosowanej** następnej akcji, co wprowadza dodatkową wariancję. Expected SARSA zastępuje ją **wartością oczekiwaną** po rozkładzie polityki: $\sum_{a'}\pi(a'\mid s')Q(s',a')$. Mniejsza wariancja celu pozwala stosować **większy krok uczący** i daje stabilniejsze, często lepsze uczenie.

Jest to metoda elastyczna: jeśli polityka, po której liczymy wartość oczekiwaną, jest tożsama z polityką zachowania, to działa **z-polityką**; jeśli przyjmiemy politykę zachłanną, Expected SARSA staje się zwykłym **Q-learningiem** (poza-polityką).

---

## Zadanie 6 - diagnoza eksplozji wartości na robocie

### (a) przyczyny

1. **Śmiertelna trójca.** Łączysz aproksymację (sieć neuronowa), bootstrapping (półgradientowa SARSA korzysta z $r+\gamma\hat v(s')$) oraz, po przeniesieniu na robota, rozjazd rozkładu danych (polityka i napotykane stany różnią się od treningowych, co działa jak element poza-polityki). To klasyczny przepis na rozbieżność estymat.
2. **Za duży efektywny krok uczący / rozjazd dystrybucji (sim-to-real).** Na prawdziwym robocie stany, skala nagród i opóźnienia różnią się od symulacji; przy bootstrappingu i niewystarczająco małym $\alpha$ pojedyncze duże błędy TD eskalują (jak przeskok krytyka w Zadaniu 1), a brak normalizacji cech to wzmacnia.

(Inne dopuszczalne: brak sieci docelowej powodujący gonienie ruchomego celu; nieergodyczność/rzadkie odwiedzanie stanów korygujących.)

### (b) środki zaradcze

- Przeciw trójcy: ogranicz jeden z jej filarów. Np. **sieć docelowa** (zamrożone parametry celu), **przejście na dane z-polityką**, albo **mniej agresywny bootstrapping** (n-krokowe zwroty zamiast TD(0)).
- Przeciw za dużemu krokowi/rozjazdowi: **mały, malejący $\alpha$**, **clipping gradientu**, **normalizacja cech i nagród**, **randomizacja domeny** w symulacji, wdrożenie w trybie cienia z monitorowaniem.

### (c) dlaczego Monte Carlo jest bezpieczniejsze

Monte Carlo uczy się z **rzeczywistych, pełnych zwrotów** $G_t$, bez bootstrappingu. Cel aktualizacji nie zależy od bieżącej (potencjalnie błędnej) estymaty wartości, więc nie powstaje dodatnie sprzężenie zwrotne, w którym zawyżona wartość jednego stanu zawyża cel dla innego. Usunięcie jednego z trzech filarów śmiertelnej trójcy (bootstrappingu) znacząco zmniejsza ryzyko rozbieżności, choć kosztem większej wariancji i konieczności posiadania epizodów.

---

## Zadanie 7 - bank pytań (zwięzłe odpowiedzi)

1. **Optymistyczne estymaty startowe:** ustawienie $Q_1$ wyraźnie powyżej realnych nagród sprawia, że każda jeszcze niewypróbowana akcja wygląda atrakcyjnie, więc agent na początku sam siebie zachęca do eksploracji, zanim estymaty opadną do prawdziwych wartości. Działa jednak słabo w problemach niestacjonarnych.
2. **Planowanie w czasie decyzji (np. MCTS):** zamiast liczyć globalną politykę dla całego modelu, skupia obliczenia na wybraniu najlepszego najbliższego ruchu, budując lokalne wartościowanie wokół bieżącego stanu. Opłaca się, gdy mamy dużo czasu na pojedynczą decyzję, a powtórka danego stanu jest mało prawdopodobna (np. szachy, Go).
3. **Inklinacja maksymalizacyjna:** branie maksimum po zaszumionych estymatach systematycznie zawyża wartości (nawet gdy prawdziwa wartość każdej akcji to 0). **Double-Q-Learning** trzyma dwie niezależne estymaty i jedną wybiera akcję, a drugą ocenia jej wartość, rozdzielając selekcję od oceny i usuwając dodatnie obciążenie.
4. **Post-stany (afterstates):** w wielu grach skutek własnego ruchu jest deterministyczny, a wiele różnych par stan-akcja prowadzi do tej samej pozycji po ruchu. Wartościowanie post-stanu pozwala te przypadki uwspólnić i uczyć się efektywniej; prawdziwą niewiadomą jest dopiero odpowiedź przeciwnika.
5. **PPO:** **clip** ogranicza iloraz istotności (stosunek nowej polityki do starej), dzięki czemu pojedyncza aktualizacja nie odbiega zbyt mocno od polityki bazowej (stabilność); człon **entropii** premiuje większą losowość polityki, podtrzymując eksplorację i zapobiegając przedwczesnemu zbieganiu do polityki deterministycznej.
6. **Gradient polityki a polityki losowe:** metody gradientu polityki parametryzują rozkład akcji wprost i mogą zbiegać do dowolnego rozkładu, w tym losowego z konkretnymi proporcjami. Metody zachłanne względem wartości akcji wybierają jedną akcję o najwyższej wartości, więc nie potrafią wyrazić sytuacji, w której optymalnie jest mieszać akcje (typowe przy aproksymacji stanu lub niepełnej obserwowalności).
