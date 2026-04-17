Aktor - co aktor robi, skrzynka, zachowanie aktora.
Krytyk - skrzynka, uczy sie oceniac, czy to co aktor robi ma sens i go nagradzac lub kara w sytuacjach, w ktorych laduje. Krytyk pozwoli nam wyciagnac wnioski wczesniej niz pozniej. DOwiadujemy sie, ze czekaja nas zle konsekwencje lub dobre wczesniej niz one wystapia.

Taki system nie uczy sie rownomiernie. Zamiast prawdziwych gradientow uzywamy batchy, stochastic. Batche rozmiaru 1, uczymy sie na takich.

Krytyk ocenia danego agenta. Aktor nie jest stabilny. Zmienia sie, zeby krytyk go pochwalil. Ale jak aktor sie zmienil, to oceny krytyka moga zrobic sie nieaktualne, a wiec nasze zachowanie znowu musi sie zmienic, musi nadazac za krytykiem.

Timeouty - dla patyka im dluzej tym lepiej, dla ladownika niekoniecznie.

akcje wybieramy w oparciu o politykę, polityka jest z daszkiem (polityka aproksymowana, pewnie jakimis neuronami), zalezy od tego w jakim jeestesmy stanie, oraz od jakich wag aktualnie opisuja nas black box z polityką. polityka zawiera prawodpodbienstwa, random choice wazony, wychodzi akcja, robimy te akcje.

liczymy blad wartosiowania w schemacie TD(0).

jest funkcja v - funkcja wartosciujaca sciany, przyjmuje stan i zwraca
aproksymacja pi to aktor, roznica miedzy oczekiwanym
robimy bootstraping , bierzemy najblizsza nagrode i zastepujemy funkcje wartoscioujaca kolejnym wywolaniem funkcji v
jak delta jest duza to krytyk byl pesymista i wyszlo lepiej niz myslal, i odwrotnie

dla aktora trzeba maksymalizowac, zeby mial duzo nagrod, dlatego funkcja dajemy minus, aktor maksmalizuje: wyliczamy prawdopodibienstwa akcji pod warunkiem stanu, nie da sie trenowac gradientowo aktora, przynajmniej tak sie wydawalo, twierdzenie o gradiencie polityki, efekt tego twierdzenia, gradientu z polityki po dostawanych nagrodach nie da sie policzyc, ale da sie policzyc wektor ktory bedzie miec taki sam kierunek i zwrot jak prawdziwy zwrot ale jego norma moze byc zupelnie inna.
delta jest na plusie, czyli akcja byla lepsza niz sie wydawalo, to poprawiamy prawdopodobienstwo tej akcji
jak wyjdzie ujemna, to obnizamy pp tej akcji

entropia politykia - 

jak liczy sie entropie w informatyce (czego, bity), jednostka informacji
