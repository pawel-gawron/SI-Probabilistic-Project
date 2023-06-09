# SI Probabilistic Project

Projekt zaliczeniowy z przedmiotu Sztuczna Inteligencja w Robotyce polegało na stworzeniu systemu śledzącego przechodniów wykorzystującego probabilistyczne modele grafowe. System ma za zadanie określić położenie przechodniów na kolejnych klatkach obrazu z kamery poprzez przypisane prostokątów ograniczających (ang. Bounding Boxes, BBoxes) do poszczególnych osób.

Wykorzystywane w tym celu są porównywania histogramów (kanały H-ue, V-alue oraz Gray) ograniczone przez zakresy Bounding Boxes. Dodatkowo obszar ten jest zawężany do połowy wysokości oraz sdzerokości w celu ograniczenia wpływu tła. Obliczone zostaję podobieństwo Bound Boxa z poprzedniego zdjęcia ze wszystkimi Bounding Boxami ze zdjęcia kolejnego. Na podstawie tego tworzony jest DiscretFactor zawierający te prawdopodobieństwa wraz z uwzględnieniem możliwości wystąpienia nowej osoby który następnie dodawany jest do Grafu.

Kolejnym krokiem jest stworzenie DiscreteFactor jako kombinacje pomiędzy wszystkimi Bounding Boxami (wymóg BeliefPropagation), lecz uwzględniając, że obiekt nie może być zakwalifikowany dwa razy jako ta sama klasa obiektu. Określane jest to poprzez macierz "nodesPossibilityMatrix" która na swojej przekątnej ma wartości 0 oprócz pierwszego elementu który symbolizuje nową klase na zdjęciu. Następnie za pomocą funkcji "combinations" tworzone są wszystkie kombinacje pomiędzy węzłami oraz dodawane są do Grafu.

Ostateczną fazą jest wykorzystanie BeliefPropagation które określa nam klase z największym prawdopodobieństwem. Należy również pamiętać o odjęciu wartości "1" od wszystkich wyników ze względu na oznaczenie nowego wystąpienia jako klasy "-1".
