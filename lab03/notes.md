# Przeszukiwanie drzewa metodą Monte Carlo (MCTS)
Monte Carlo Tree Search

Składa się z czterech kroków: selekcji (selection), rozwinięcia (expansion), symulacji (simulation/rollout) i propagacji wstecznej (backpropagation).

# Raport

Zaimplementowałem wymagane etapy algorytmu MCTS. W przypadku selekcji, wartość pola `value` w dzieciach, to tak naprawdę wartościowanie stanu przeciwnika, dlatego brałem, podczas liczenia UCB, wartość wyrażenia $(1-value)$: `(1 - child.value) + self.c_coefficient * math.sqrt(math.log(self.times_chosen) / child.times_chosen)`. Cały kod znajduje się w pliku `isolation.py`.

Po pierwszych uruchomieniach widać, że gracz MCTS miaźdzy do zera przeciwnika losowego, co już świadczy o nieco bardziej intetligentnych ruchach. Przeprowadziłem kilka testów wyników gry między graczami MCTS, różniącymi się wartościami parametrów `time_limit`, czy `c_coefficient`. Różne, przykładowe wywołania testowe umieszczałem w osobnych funkcjach, oznaczanych, jako `ex01()`, `ex02()` itd.

Funkcje do wizualizacji zostały umieszczone w pliku `visualizations.py`.
