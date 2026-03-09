zum starten der Docker bitte folgende befehle in der Powershell ausführen:

Um einmal alle vorherigen docker zu entfernen und fehler vorzebeugen: docker system prune -a --volumes

dann in das verzeichnis gehen wo man den docker liegen hat (mit cd)

Damit der docker nicht abstürtzt wenn man alle docker gleichzeitig startet muss man eine verzögerung einbauen: 

docker-compose up --build face-extraction; Start-Sleep -s 10; docker-compose up --build deepface-app; Start-Sleep -s 10; docker-compose up --build moondream;

so verhindert man das die docker beim starten abstürzen