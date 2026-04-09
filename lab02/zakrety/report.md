## Implementacja SARSA

Zaimplementowałem funkcje do liczenia n-krokowego zwrotu zdyskontowanego oraz importance sampling ratio:

```python
# G - n-krokowy zwrot zdyskontowany
def _return_value(self, update_step):
    return_value = 0.0

    t = update_step
    end = min(t + self.step_no, self.final_step)
    for i in range(t + 1, end + 1):
        return_value += self.discount_factor ** (i - t - 1) * self.rewards[self._access_index(i)]
    if t + self.step_no < self.final_step: # bootstraping
        s = self.states[self._access_index(t + self.step_no)]
        a = self.actions[self._access_index(t + self.step_no)]
        return_value += self.discount_factor ** self.step_no * self.q[s, a]
    return return_value

# p - importance sampling ratio
def _return_value_weight(self, update_step):
    return_value_weight = 1.0

    t = update_step
    end = min(t + self.step_no, self.final_step)
    for i in range(t + 1, end):
        s_i = self.states[self._access_index(i)]
        a_i = self.actions[self._access_index(i)]
        actions = available_actions(s_i)

        pi = self.greedy_policy(s_i, actions)[a_i]
        b = self.epsilon_greedy_policy(s_i, actions)[a_i]
        return_value_weight *= pi / b
    return return_value_weight
```

## Corner B
Algorytm z sukcesem uczy się przejazdu przez trasę `corner_b`. Poniżej przedstawiam początkowe epizody (długie przejazdy) oraz te końcowe (z używaniem tylko optymalnej strategii, dodałem na końcu przeprowadzenie kilku ewaluacyjnych epizodów).

(plot)[plots_b/track_150.png]
(plot)[plots_b/track_750.png]
(plot)[plots_b/track_29600.png]

Podczas początkowych prac pojawił się problem, że samochód przejeżdżał przez sciany, ponieważ sprawdzał tylko, czy `car.next_position()` znajduje się na mapie, ale nie, czy piksele przez, które przechodzi też stanowią drogę. Narzędzia sztucznej inteligencji poradziły, by użyć prostego algorytmu sprawdzającego, czy na tej lini znajduje się przeszkoda, np. algorytm `DDA Line Drawing Algorithm`.

## Corner C

`corner_c` był już wiekszym problemem dla algorytmu. Pierwotna nauka trwała bardzo długo, zanim tak naprawdę algorytm odkrył drogę do mety - meta znajduje się znacznie dalej. Musiałem też na wybrać inne parametry, ustawiłem `step_no=5` i `experiment_rate=0.1`. Dla takich parametrów, cały trening (30000 epizodów) trwał około 40 minut. Oto przykładowe trasy, od początkowch, po optymalne. Widać też sporo powrotów na start, po uderzeniach w ścianę.

(plot)[plots_b/track_100.png]
(plot)[plots_b/track_1600.png]
(plot)[plots_b/track_29750.png]