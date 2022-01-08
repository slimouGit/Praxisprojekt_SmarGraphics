# Praxisprojekt_SmarGraphics

Das in diesem Repository abgelegte Projekt beinhaltet eine Anwendung zu der Klassifizierung von Abbildungen mit den Ziffern Null bis Neun.
Die Anwendung wurde im Rahmen der Semesterarbeit im Seminar Smart Graphics des Wintersemesters 2021 entwickelt.

Enthalten sind folgende 4 Klassen und 2 Verzeichnisse:

- config.py: enthält global zu konfigurierende Attribute der Anwendung
- dataset_initializer.py: zur Erstellung eines Test- und Trainingsdatensatzes
- digit-classifier.py: zur Klassifikation von Bilddaten
- image_converter: zum Konvertieren von Abbildungen in maschinenlesbare Biddaten (wird aus digit-classifier.py aufgerufen)	
- im Verzeichnis data liegen die mit der Klasse dataset_initializer.py erstellten Test- und Trainingsdatensätze 
- im Verzeichnis image liegen Beispieldaten von Ziffern zum Klassifizieren ab und die Daten zum Erstellen der Test- und Trainingsdatensätze 

