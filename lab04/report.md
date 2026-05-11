

Problem

```
Bug aktywuje się tylko gdy truncated=True (bo wtedy niesłusznie zerujemy V(s')). Pytanie: jak często ta ścieżka się odpala?

LunarLander-v3 kończy się przez terminated=True w czterech głównych przypadkach:

lądowanie (sukces)
rozbicie
wyleciał poza ekran
całkowite zatrzymanie na ziemi (oba kontaktery dotykają podłoża)
truncated=True odpala się dopiero po 1000 krokach bez żadnego z powyższych — czyli gdy lądownik zawisa w powietrzu i nie podejmuje decyzji.

Co to znaczy w praktyce:

Wytrenowany agent ląduje w 200-400 krokach → terminated → bug się nie odpala.
Średnio wytrenowany rozbija się szybko → terminated → bug się nie odpala.
Świeży agent często rozbija się w pierwszych 100-200 krokach (silniki włączone losowo, paliwo się kończy, leci w bok) → terminated → bug nie odpala.
Jedyna sytuacja truncated: agent znajdzie patologiczną politykę zawisania na silniku głównym przez 1000 kroków — to rzadkość, bo używanie silnika kosztuje (-0.3 reward/krok), więc gradient szybko od tego odciąga.
W CartPole jest dokładnie odwrotnie: najlepsza możliwa polityka prowadzi WPROST do truncation (idealny balans = 500 kroków = truncated). Bug uderza w 100% epizodów które agent rozegrał perfekcyjnie. Dlatego w CartPole rośnie "ścianka" — im lepiej gra, tym mocniej go karzesz, aż się zacina.

Krótko: bug jest groźny proporcjonalnie do tego jak często ścieżka sukcesu kończy się przez truncation. CartPole — zawsze. LunarLander — prawie nigdy.
```