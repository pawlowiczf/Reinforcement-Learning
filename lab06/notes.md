# SAC - Soft Actor-Critic

SAC to algorytm uczenia ze wzmocnieniem klasy **off-policy**, łączący podejście aktor-krytyk z zasadą **maksymalizacji entropii**. Dzięki temu agent nie tylko dąży do wysokich nagród, ale jest też aktywnie zachęcany do eksploracji.

---

## Aktor i krytyk

### Aktor - polityka $\pi_\theta(s)$

Aktor reprezentuje **politykę agenta**. Nie zwraca jednak pojedynczej akcji - modeluje **rozkład prawdopodobieństwa akcji** dla danego stanu:

$$\pi_\theta(s) \rightarrow \mathcal{N}(\mu, \sigma)$$

W praktyce sieć aktora przewiduje średnią $\mu$ i odchylenie standardowe $\sigma$, z których próbkuje się akcję. Dzięki temu eksploracja jest **wbudowana w politykę**, a nie dodawana z zewnątrz (jak np. szum Ornsteina-Uhlenbecka w DDPG).

### Krytyk - funkcja wartości $Q_\phi(s, a)$

Krytyk ocenia, **jak dobra była dana akcja w danym stanie**. Jest to zadanie trudniejsze niż rola aktora - krytyk musi nauczyć się przewidywać skumulowaną przyszłą nagrodę dla dowolnej pary $(s, a)$.

---

## Jak współdziałają sieci

Aktor i krytyk trenują się **nawzajem**:

- Aktor chce maksymalizować ocenę krytyka, dlatego dostarcza krytykowi akcje, a w zamian otrzymuje od niego **gradienty** niezbędne do własnej aktualizacji.
- Krytyk uczy się z zebranych doświadczeń $(s, a, r, s')$, oceniając na ile wybory aktora były trafne.

Obie sieci gonią się nawzajem - ich gradienty są **konkurujące**, co czyni trening delikatną równowagą.

---

## Cel: maksymalizacja entropii

Standardowe RL maksymalizuje tylko sumę nagród. SAC rozszerza ten cel o **entropię polityki**:

$$J(\pi) = \sum_t \mathbb{E}\bigl[r(s_t, a_t) + \alpha \cdot H(\pi(\cdot \mid s_t))\bigr]$$

| Symbol | Znaczenie |
|---|---|
| $r(s, a)$ | nagroda za wykonaną akcję |
| $H(\pi(\cdot \mid s))$ | entropia polityki (miara jej losowości) |
| $\alpha$ | temperatura - waga entropii względem nagrody |

Wysoka entropia oznacza politykę bardziej eksplorującą. Agent jest nagradzany za **różnorodność zachowań**, o ile nie odbywa się to kosztem nagród.

---

## Funkcje strat

### Strata aktora

Aktor minimalizuje (lub równoważnie - maksymalizuje ujemność):

$$L_\pi = \mathbb{E}_{s, a \sim \pi}\bigl[\alpha \log \pi(a \mid s) - Q(s, a)\bigr]$$

Interpretacja: chcemy akcji, które **krytyk wysoko ocenia** ($-Q$), ale jednocześnie **zwiększają entropię** polityki ($\alpha \log \pi$). Są to dwa sprzeczne siły - ich balans reguluje $\alpha$.

### Strata krytyka

Krytyk uczy się przez minimalizację błędu Bellmana:

$$L_Q = \mathbb{E}\bigl[(Q(s, a) - y)^2\bigr]$$

gdzie wartość docelowa $y$ to:

$$y = r + \gamma \bigl(\min(Q_1', Q_2')(s', a') - \alpha \log \pi(a' \mid s')\bigr)$$

Człon $-\alpha \log \pi$ to kara entropii w wartości docelowej - sprawia, że krytyk uwzględnia premię za eksplorację.

---

## Dwa krytycy: problem przeszacowania

Krytyk popełnia błędy, a aktor szybko zaczyna je **wykorzystywać na swoją korzyść** - wybiera akcje, które krytyk błędnie ocenił zbyt wysoko. To prowadzi do niestabilności treningu.

Rozwiązanie: **dwie niezależne sieci krytyka** $Q_1$, $Q_2$. Przy wyznaczaniu wartości docelowej bierzemy **minimum** ich ocen:

$$Q_{target}(s', a') = \min\bigl(Q_1(s', a'),\ Q_2(s', a')\bigr)$$

Nawet jeśli jedna sieć przeszacuje wartość, druga ją "hamuje". Technika ta nosi nazwę **Clipped Double-Q** i pochodzi z algorytmu TD3.

---

## Sieć docelowa (target network)

Do obliczania wartości docelowej $y$ używana jest **osobna, wolno aktualizowana kopia krytyków** - tzw. sieć docelowa $Q'$.

Aktualizuje się ją nie przez gradient, lecz przez **wykładniczą średnią kroczącą** (EMA) wag:

$$\phi' \leftarrow \tau \cdot \phi + (1 - \tau) \cdot \phi'$$

gdzie $\tau \ll 1$ (np. $0.005$). Dzięki temu wartości docelowe zmieniają się powoli, co **stabilizuje trening** - krytyk nie goni za ciągle zmieniającym się celem.

---

## Automatyczna regulacja temperatury $\alpha$

Zamiast ręcznie dobierać temperaturę $\alpha$, SAC potrafi **uczyć się jej automatycznie**. Definiujemy docelową entropię $\mathcal{H}_{target}$:

$$\mathcal{H}_{target} = -\dim(\mathcal{A})$$

np. dla przestrzeni akcji o 6 wymiarach: $\mathcal{H}_{target} = -6$.

Strata temperatury:

$$L_\alpha = \mathbb{E}_{a \sim \pi}\bigl[-\alpha \cdot (\log \pi(a \mid s) + \mathcal{H}_{target})\bigr]$$

Jeśli entropia polityki jest za niska (agent za mało eksploruje), $\alpha$ rośnie. Jeśli za wysoka (agent chaotyczny), $\alpha$ maleje.

---

## Podsumowanie architektury

```
Replay Buffer  <---  środowisko (doświadczenia: s, a, r, s')
       |
       v
  Minibatch
  /        \
Aktor       Krytyk (Q1, Q2)
  |              |
  |     min(Q1', Q2') - sieć docelowa (EMA)
  |              |
  L_π          L_Q
  |              |
gradient     gradient
  \              /
   aktualizacja wag
```

---

## Kluczowe właściwości SAC

| Właściwość | Opis |
|---|---|
| Off-policy | Uczy się z replay buffera - efektywne próbkowanie |
| Eksploracja przez entropię | Wbudowana w politykę, nie wymaga zewnętrznego szumu |
| Stabilność | Dwa krytycy + sieć docelowa zapobiegają przeszacowaniu |
| Automatyczne $\alpha$ | Brak konieczności ręcznego tuningu temperatury |
| Ciągłe przestrzenie akcji | Zaprojektowany z myślą o robotyce i symulacjach fizycznych |