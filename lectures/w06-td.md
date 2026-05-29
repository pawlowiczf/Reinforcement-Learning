# Uczenie oparte o różnice czasowe
Temoral Difference Learning
np. w SARSA, jako bootstraping

Algorytm TD(0) jest stosunkowo podobny do Monte Carlo (wykład wcześniejszy).

### TD(0) for estimating $v_\pi$

$$
\delta_t = R_{t+1} + \gamma V(S_t+1) - V(S_t)
$$

###

Zalety metod TD:
- nie interesuje nas model środowiska (jak wybierane są akcje, środowiska)
- gwarancja zbieżności, jak MC
- nie wymaga epizodów (bo aktualizujemy wartościowanie po każdym kroku)

Wartość stanu:
$$
V(S_t) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots] \\
V(S_t) = \mathbb{E}[R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + \ldots)] \\
V(S_t) = \mathbb{E}[R_{t+1} + \gamma V(S_{t+1})]
$$

W praktyce $V(S_t)$ jest tylko naszym szacunkiem, który może być błędny. Skoro równanie Bellmana musi zachodzić dla prawdziwych wartości, to możemy sprawdzić o ile nasz szacunek jest niespójny:
$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$
Skoro wiemy, o ile się mylimy, to poprawiamy szacunek w tym kierunku o krok $\alpha$:
$$
V(S_t) = V(S_t) + \alpha \cdot \delta_t
$$

### Regularyzacja entropii
Zamiast maksymalizować tylko nagrodę, agent maksymalizuje nagrodę plus entropię swojej polityki.

$$
J(\pi) = \mathbb{E} \left[ \sum_t r_t + \alpha \cdot \mathbb{H} \bigl( \pi (\cdot|s_t) \bigr) \right]
$$
gdzie $\mathbb{H}=-\sum_a \pi (a|s)\log \pi(a|s)$ to entropia polityki, a $\alpha$ to temperatura. Entropia to liczba, która mówi jak bardzo rozproszony jest rozkład prawdopodobieństwa nad akcjami w danym stanie. Gdy entropia jest równ zero, to polityka jest deterministyczna - agent zawsze wybiera tę samą akcję, np. $[0, 0, 1, 0]$. Gdy entropia osiąga wartość maksymalną, to polityka jest jednostajna - agent traktuje wszystkie akcje jako równie dobre, np. $[0.25, 0.25, 0.25, 0.25]$.

Dzięki dodaniu entropii zyskujemy:
- agent nie zbiega przedwcześnie do lokalnego optimum. Jeśli jedna akcja jest trochę lepsza, nie ignoruje całkowicie pozostałych
- agent zachowuje eskplorację bez potrzeby sztucznych mechanizmów, np. e-greedy
- polityka jest bardziej odporna np. przy zmianie środowika

```
for każdy epizod do
    S ← początkowy stan

    for każdy krok do
        wybierz A w oparciu o π̂ (· | S, w)
        wykonaj akcję A
        zaobserwuj nagrodę R i nowy stan S'
        policz błąd wartościowania w oparciu o schemat TD(0):

        if S' nie jest stanem terminalnym then
            δ ← R + γv̂(S', w) − v̂(S, w)
        else
            δ ← R − v̂(S, w)
        end if

        policz stratę dla krytyka L_critic ← δ²
        policz stratę dla aktora L_actor ← −δ · log π̂(A | S, w)
        policz bonus entropii L_entropy ← −β · H(π)
        policz łączną stratę L ← L_critic + L_actor + L_entropy

        zaktualizuj wagi spadkiem po gradiencie w ← w − α∇wL
        S ← S'
    end for
end for
```
### Q-learning
Off-policy temporal difference control. Bezpośrednio aproksymuje optymalną politykę, niezależnie od tej, która obecnie wpływa na zachowanie agenta.

### Sarsa z wartościami oczekiwanymi

Stosować, gdy liczba akcji do wyboru w danym stanie nie jest zbyt duża, bo pod tych akcjach musimy całkować.

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \mathbb{E}[Q(S_{t+1}, A_{t+1}) \mid S_{t+1}] - Q(S_t, A_t) \right]
$$
$$
\leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right]
$$

### Problem inklinacji w kierunku maksymalizacji
(maximization bias)

### Double Q-learning
Q learning jest off-policy.

Problem: te same próbki są używane zarówno do wyboru najlepszej akcji, jak i do oceny jej wartości. Szacunki $Q(B,a)$ są zaszumione. Gdy używamy ich do wyboru akcji, celowo szukamy tej z największym szumem. A potem używamy tego samego zaszumionego szacunku, jako wartości - dostajemy systematycznie zawyżoną ocenę.

Rozwiązanie: podzielić odpowiedzialność i przechowywać dwie niezależne estymaty.

$$
Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q_2(S_{t+1}, \argmax_a Q_1(S_{t+1}, a)) - Q_1(S_t, A_t) \right]
$$

Problem z Q-learningiem: ten sam Q robi dwie rzeczy na raz - wybiera akcję oraz ocenia. Jeśli Q przypadkowo zawyży wartość jakiejś akcji, to później `argmax` ją wybierze, błąd się wzmocni.

Nowy proces:
1. Oceniamy politykę, taką jaka byłaby w oparciu o Q1. Mamy starą estymatę Q1 i robimy nową.
2. Bootstraping robimy w oparciu o Q2

