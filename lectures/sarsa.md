## SARSA
State -> ACtion -> Reward -> State -> Action

b - behaviour policy - polityka, według której agent faktycznie działa (eksploruje)
π - target policy - polityka, której agent się uczy (zazwyczaj greedy względem Q)

Intuicja: Jeśli π i b wybierałyby te same akcje → ρ = 1 (brak korekty). Jeśli π woli akcję, którą b rzadko wybiera → ρ > 1 (wzmocnienie). Jeśli π nigdy by tej akcji nie wybrała → ρ = 0 (ignorujemy tę trajektorię).