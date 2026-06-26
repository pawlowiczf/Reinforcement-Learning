# Uczenie ze Wzmocnieniem - Egzamin 1 (zestaw ćwiczeniowy)

> Zestaw przygotowany na wzór egzaminu zerowego i materiału wykładowego.
> Pokrywa pięć kategorii: **liczenie**, **teoria**, **propozycja rozwiązania problemu**, **quiz/wiedza** oraz **myślenie**.
> Sugerowany czas: ok. 90 minut. Maksimum: **20 pkt** (zadania 1-6) + 2 pkt bonus.
>
> Rozwiązania znajdziesz w osobnym pliku `UzW_Egzamin_1_rozwiazania.md`.

---

## [4 pkt] Zadanie 1 - z liczenia (ręczny krok uczący Aktor-Krytyk)

Zasymuluj na kartce przebieg jednego kroku uczącego dla następującego agenta:

- agent korzysta z najbardziej podstawowego schematu **Aktor-Krytyk** (A2C, *Advantage Actor-Critic*);
- obserwowany stan środowiska to dwie współrzędne: $x$ oraz $y$;
- w każdym stanie możliwe jest wykonanie jednej z dwóch akcji: $a_0$ albo $a_1$;
- polityka agenta dana jest wzorami:

$$\pi_\theta(A=a_0 \mid s) = \frac{\theta_a x}{\theta_a x + \theta_b y}, \qquad \pi_\theta(A=a_1 \mid s) = \frac{\theta_b y}{\theta_a x + \theta_b y}$$

- krytyk opisany jest funkcją $v_w(s) = w\,(x + y)$;
- agent rozpoczął krok w stanie $s_1 = [3,\,1]$, następnie wykonał akcję $a_1$;
- w efekcie tej decyzji znalazł się w stanie $s_2 = [1,\,1]$ oraz otrzymał nagrodę $r = 10$;
- początkowe parametry polityki to $\theta_a = 2,\ \theta_b = 2$;
- konfiguracja krytyka to $w = 2$, współczynnik uczenia $\alpha = 0.5$, **nie korzystamy z discount factor** ($\gamma = 1$);
- ten sam błąd TD pełni rolę przewagi (*advantage*) zarówno przy aktualizacji krytyka, jak i aktora.

**Polecenie:** Pokaż jak zmienią się aktor i krytyk po jednym kroku uczenia. Jak zmiany te wpłyną na późniejsze preferencje agenta? Skomentuj, czy dobrany współczynnik $\alpha$ wydaje się bezpieczny.

---

## [4 pkt] Zadanie 2 - z liczenia (wielorękie bandyty: selekcja i aktualizacja)

Rozważamy problem **3-rękiego bandyty**. Po pewnej liczbie prób mamy następujący stan wiedzy:

| akcja | liczba wyborów $N(a)$ | aktualna estymata $Q(a)$ |
|:-----:|:---------------------:|:------------------------:|
| $a_1$ | 1                     | 0.8                      |
| $a_2$ | 8                     | 1.0                      |
| $a_3$ | 1                     | 0.5                      |

Łączna liczba dotychczasowych kroków to $t = 10$.

**(a)** Którą akcję wybierze polityka **zachłanna**, a którą wybierze **$\varepsilon$-zachłanna** w trybie eksploracji? Jaki ułamek wyborów (przy $\varepsilon = 0.3$) trafi w akcję $a_2$ przy losowaniu spośród wszystkich akcji?

**(b)** Wyznacz wybór akcji metodą **UCB1** dla wzoru

$$A_t = \arg\max_a \left[\, Q(a) + c\sqrt{\frac{\ln t}{N(a)}}\,\right], \qquad c = 1.$$

Skomentuj, dlaczego wynik różni się od wyboru zachłannego.

**(c)** Zastosuj **gradientowego bandytę** (selekcja soft-max po preferencjach $H$). Załóż początkowe preferencje $H = [0,\,0,\,0]$, średnią nagrodę dotychczas $\bar{R} = 0.5$ oraz krok uczący $\alpha = 0.5$. W bieżącym kroku wybrano akcję $a_2$ i uzyskano nagrodę $R = 1$. Podaj nowe preferencje oraz nowy rozkład prawdopodobieństwa wyboru akcji. Reguła aktualizacji:

$$H(a) \leftarrow H(a) + \alpha\,(R - \bar{R})\,\big(\mathbb{1}[a = A_t] - \pi(a)\big).$$

**(d)** Wariant **bernoulliowski + próbkowanie Thompsona**. Dla pewnej akcji przyjęto a priori rozkład $\mathrm{Beta}(1, 1)$. Zaobserwowano 3 sukcesy i 2 porażki. Podaj rozkład a posteriori i jego wartość oczekiwaną oraz wyjaśnij, jak próbkowanie Thompsona wykorzysta ten rozkład przy wyborze akcji.

---

## [3 pkt] Zadanie 3 - z teorii

**(a)** Zapisz **równanie Bellmana** dla funkcji wartościującej stany $v_\pi$ oraz **równanie optymalności Bellmana** dla $v_*$. Wyjaśnij rolę każdego składnika i wskaż dokładnie, który element znika przy przejściu od $v_\pi$ do $v_*$ i dlaczego.

**(b)** Porównaj **zwyczajne** i **ważone** próbkowanie istotności (*ordinary / weighted importance sampling*) w kategoriach **obciążenia** i **wariancji**. Który estymator jest obciążony? Który ma potencjalnie nieograniczoną wariancję? Dlaczego obecność pętli w MDP pogarsza sytuację?

**(c)** Wyjaśnij pojęcie **śmiertelnej trójcy** (*the deadly triad*). Wymień jej trzy składniki i wyjaśnij, na minimalnym przykładzie dwóch stanów wartościowanych jako $w$ oraz $2w$, dlaczego ich połączenie grozi rozbieżnością estymat.

---

## [3 pkt] Zadanie 4 - propozycja rozwiązania problemu (zadanie otwarte)

Masz za zadanie przygotować inteligentnego agenta sterującego **grupą wind w wieżowcu** (np. 6 wind, 40 pięter). Celem jest minimalizacja średniego czasu oczekiwania pasażerów i czasu przejazdu. Natężenie ruchu zmienia się w ciągu dnia (poranny szczyt w górę, popołudniowy w dół). Agent zna wezwania z paneli na piętrach oraz przyciski wciśnięte wewnątrz kabin, ale nie zna z góry, dokąd zmierza dany pasażer ani ilu ich jest.

**(a)** Zaprojektuj (wysokopoziomowo) sposób rozwiązania tego problemu. Omów przyjęte założenia: sformułowanie jako MDP (stan, akcje, nagroda), wybór algorytmu, sposób reprezentacji/aproksymacji, eksplorację, organizację treningu (np. symulacja).

**(b)** Uzasadnij swoje wcześniejsze decyzje: jakie argumenty przemawiają za takimi wyborami? Przedstaw przynajmniej trzy z nich.

**(c)** Przeanalizuj ryzyka związane z przedstawionym projektem: co może mieć największy wpływ na jego niepowodzenie? Jak zminimalizować prawdopodobieństwo takiego scenariusza? Skup się na jednym takim czynniku.

---

## [3 pkt] Zadanie 5 - quiz (wykazanie się wiedzą)

**(a)** Jak zmiana estymacji ze **średniej z próbek** na **średnią wykładniczo ważoną aktualnością** (stałe $\alpha$) wpłynie na zachowanie agenta w środowisku **niestacjonarnym**? Czy spełnione będą standardowe warunki zbieżności $\sum_n \alpha_n = \infty$ oraz $\sum_n \alpha_n^2 < \infty$?

**(b)** Jak liczba kroków planowania $n$ na jeden krok rzeczywisty w algorytmie **Dyna-Q** wpływa na szybkość uczenia i koszt obliczeniowy? W jakiej sytuacji większe $n$ pomaga najbardziej?

**(c)** Dlaczego **SARSA z wartościami oczekiwanymi** (*Expected SARSA*) bywa skuteczniejsza od zwykłej SARSY przy tym samym kroku uczącym? Czy jest to metoda z-polityką czy poza-polityką?

---

## [3 pkt] Zadanie 6 - z myślenia (diagnoza)

Trenujesz robota w **symulacji**, korzystając z półgradientowej SARSY i **sieci neuronowej** jako aproksymatora funkcji wartościującej. W symulacji agent uczy się świetnej polityki, jednak po wgraniu na **prawdziwego robota** estymaty wartości w ciągu kilku minut rozbiegają się do nieskończoności.

**(a)** Wskaż przynajmniej **dwie** prawdopodobne przyczyny tego zjawiska, wynikające wprost z teorii omawianej na wykładzie.

**(b)** Dla każdej przyczyny zaproponuj **konkretny** środek zaradczy.

**(c)** Krótko uzasadnij, dlaczego ten sam problem byłby mniej prawdopodobny, gdyby zamiast bootstrappingu TD użyć pełnych zwrotów Monte Carlo.

---

## [+2 pkt bonus] Zadanie 7 - krótki bank pytań do przemyślenia

Odpowiedz zwięźle (1-3 zdania) na trzy dowolne z poniższych:

1. Po co stosuje się **optymistyczne startowe estymaty wartości** i jaki ma to związek z eksploracją?
2. Czym różni się **planowanie w czasie decyzji** (np. MCTS) od budowania pełnej polityki dla całego modelu? Kiedy to pierwsze się opłaca?
3. Co to jest **inklinacja w kierunku maksymalizacji** (*maximisation bias*) i jak **Double-Q-Learning** ją ogranicza?
4. Dlaczego w wielu grach planszowych korzystniejsze jest wartościowanie **post-stanów** (*afterstates*) niż par stan-akcja?
5. Jaką rolę pełni **clip** oraz człon **entropii** w funkcji celu algorytmu **PPO**?
6. Czemu metoda **gradientu polityki** potrafi nauczyć się **losowej** polityki optymalnej, a metody zachłanne względem wartości akcji nie?
