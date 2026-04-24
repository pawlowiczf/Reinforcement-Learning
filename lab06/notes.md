### SAC - Soft Actor Critic
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

krytyk popelnil blad i aktor zaczyna to wykorzystywac na swoja korzysc, co jest złe
$Q(s,a) -> Q_1, Q_2$ - są dwie funkcje Q i z nich bierzemy minimum.

do treningu aktora wybieramy wygładzone Q, trzecia sieć, średnia przeszłych wartości krytyków

### HER - Experience Rate

Mamy binarne rewardy - wyobraźmy sobie - albo się udało, albo nie. Moze zdażyć się, że taki trening będzie długo trwał, bo będzie eksplorować, ale nigdy nie trafi na nagrodę. Mówisz wtedy mu - dobra nie udało Ci się to co chciałem, ale coś zrobiłeś, to uznajmy, że o to mi chodziło. Bede robil tak, ze mimo ze etap nie zakonczyl sie sukcesem, to bede tak tratkowac. W ten sposob dodaje sygnaly. Daje agentowi

polityka $\pi (a|s,g)$, gdzie g to goal. musze powiedziec polityce do jakiego celu. bede mogl mowic np. gdzie umiescic, przeniesc przedmiot, przenies tu, teraz tu itd.

### Gymnasium
Multi-Goal API
Każda obserwacja to słownik z trzema elementami.

OpenAI Soft Actor Critic - SpinningUp