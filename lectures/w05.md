# Metody Monte Carlo

Środowisko jest często skomplikowane, ciężkie do opisania matematycznie. Traktujemy je raczej jako `black box`.

Monte Carlo dla problemów Markova.

1. Opieramy się na kompletnch, ukończonych epizodach.

### Predykcja wartościowania stanu w oparciu o pierwsze wizyty i podejście MC
First-visit MC prediction, for estimating $V \approx v_\pi$.

Idziemy po epizodzie od końca do początku i sumujemy kolejne nagrody (zwrot). Będąc w czasie $t$, jeśli stan $s_t$ nie pojawia się we wcześniejszym ciągu $s_0, s_1, \ldots, s_{t-1}$, to dodaj całkowity zwrot do tablicy dla stanu $s_t$.

### Predykcja wartościowania akcji Q(s,a)

Gdy model nie jest znany, to wartościowanie stanów ma niewielką użyteczność - nie wiemy, czy dana akcja doprowadzi nas do korzystnej sytuacji.

Jeżeli polityka nigdy nie wykonała akcji $a$, to nie mamy żadnej wiedzy o jej konsekwencjach - można wtedy wykorzystyać tzw. eksplorujące otwarcie.

### Sterowanie oparte o podejście MC

Podobny schemat algorytmu, jak wyzej, ale teraz każdy epizod startujemy od losowego ($s_0,a_0$), przy czym wszystki takie pary mają niezerowe prawdopodobieństwo wystąpienia (exploring starts). Iterujemy po takim epizodzie i jeśli para ($s_t, a_t$) nie występuje w ciągu wcześniej, to aktualizujemy:
$$
Returns(s,a) \larr Returns(s,a) + G \\
Q(s,a) \larr average(Returns(s, a)) \\
\pi (s) \larr \argmax _a Q(s, a)
$$

### Predykcja wartościowania poza-polityką (off-policy)

Chcemy estymować $v_\pi$ lub $q_\pi$, ale mamy do dyspozycji tylko epizody oparte o $b \neq \pi$. $b$ to przykładowo politka eksploracyjna ($\epsilon$-greedy), a $\pi$ to polityka zachłanna.

$$
\mathbb{E} [G_t | S_t] = v_b(S_t) \\
\to \\
\mathbb{E} [\rho_{t:T-1} G_t | S_t] = v_\pi(S_t)
$$