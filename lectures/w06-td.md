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

### Q-learning
