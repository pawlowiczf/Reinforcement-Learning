Chcemy, żeby LLM robił coś innego. LLM wybiera tokeny, prawdopodobienstwo każdego. Jest to zwykly RL. Wiec mozna aplikowac dowolny aspekt RLowy do LLMa. Policy Gradient. Fine tuning na trezch poziomach: 
1. bierzemy siec, zamrazamy ja, robimy klona klona finetunujemy, output danych warstw to suma par, klon na starcie jest caly zerowy, na starcie nic nei sumuje. , inferencja sieci
2. lora adaptation, low rank adaptation, patch notes rezydualne, bazowy model + adapter

bedziemy trenowac ten adapter metoda RL, z nagrodami. Z paperu: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. Proximal Policy Optimization Algorithms.