Chcemy, żeby LLM robił coś innego. LLM wybiera tokeny, prawdopodobienstwo każdego. Jest to zwykly RL. Wiec mozna aplikowac dowolny aspekt RLowy do LLMa. Policy Gradient. Fine tuning na trezch poziomach: 
1. bierzemy siec, zamrazamy ja, robimy klona klona finetunujemy, output danych warstw to suma par, klon na starcie jest caly zerowy, na starcie nic nei sumuje. , inferencja sieci
2. lora adaptation, low rank adaptation, patch notes rezydualne, bazowy model + adapter

bedziemy trenowac ten adapter metoda RL, z nagrodami. Z paperu: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. Proximal Policy Optimization Algorithms.

Skoro za błędną faktograficzną odpowiedź dostaje 0 z korektności tak czy siak, to najlepszą strategią staje się odmowa zbudowana z samych częstych słów — zgarnia maksimum z reward_vocab i nic nie traci na korektności. Model nauczył się, że „gdy nie wiem, recytuj wysoko-słownikowy boilerplate". I ta strategia rozlała się nawet na pytania, które wcześniej umiał (Mona Lisa, Romeo i Julia) — bo to lokalnie opłacalny wzorzec.

Trzy najważniejsze wnioski
Odmowy padły dokładnie zgodnie z teorią. Szablon „I'm sorry…" miał korektność 0 → po bramkowaniu vocab = 0 → przestał się opłacać. Wszystkie trzy odmowy z 000004 zniknęły.

Powtórzenia zniknęły BEZ kary 3-gramowej. To kluczowe odkrycie: Jupiter przestał się zapętlać, mimo że reward_no_repetition było wyłączone. Czyli powtarzanie to był objaw farmienia vocab (dorzucanie pospolitych fraz), a nie osobny problem. Leczyliśmy skutek (3-gramy), a wystarczyło usunąć przyczynę (bramkowanie). Dobrze, że to wyizolowaliśmy.

Lanie wody zniknęło tam, gdzie było pustym wypełniaczem (stolica, Jupiter, złoto), a zostało tam, gdzie to realne wyjaśnienie (słońce, niebo, liście — wciąż długie, ucięte na 80 tokenach). To dobry znak: model skraca śmieci, ale nie tnie treści.

Bramkowanie nagrody za słownictwo korektnością. reward_vocab zwraca punkty za pospolite słowa tylko gdy odpowiedź jest trafna (_correctness_score > 0), inaczej 0. Dzięki temu odmowy, powtórzenia i lanie wody — które miały korektność 0 — przestały dawać darmowe punkty. Wyciągnęliśmy też wspólny pomocnik _correctness_score, a karę 3-gramową zostawiliśmy wyłączoną (okazała się zbędna)