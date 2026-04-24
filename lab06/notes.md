SAC - Soft Actor Critic
Jest to uczenie off-policy.

Aktor chce zadowolić krytyka, a krytyk krytykować aktora.
Aktor reprezentuje politykę $\pi(s) -> a$, a krytyk reprezentuje $Q(s,a)$ - ocena akcji w danym stanie (to zadanie jest trudniejsze).

Oboje się trenują. Żeby trenować politykę, potrzebny jest input od krytyka (dostarcza gradienty). Krytyk uczy się z doświadczenia (z wyborów akcji, stanów).
Jedna sieć goni drugą sieć, chociaż są to konkurujące ze sobą gradienty.

**To dla aktora:**
Polityka nie zwraca pojedynczej akcji, tylko modeluje rozkład prawdopodobieństwa akcji nad stanem.
Chcemy maksymalizować przyszłe nagrody, ale maksymalizująć entropię. Czyli zostań losową polityką (eksploracja?), ale dawaj dużo nagród. Eksploracja jest wbudowana w politykę.

$L=\alpha H - Q$ - to optymalizujemy? H - entropia.

**To dla krytyka:**
$L=(Q-y)^2$ - uczymy go różnicą tym co zwraca, a tym co powinien zwracać
$y=r + x(Q' - \alpha $

$Q(s,a) -> Q_1, Q_2$ - są dwie funkcje Q i z nich bierzemy minimum.