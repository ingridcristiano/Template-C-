#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        // =========================================================================
        // 1. CARICAMENTO SINGOLA IMMAGINE
        // =========================================================================
        // ---> MODIFICA QUESTA RIGA CON IL TUO PERCORSO <---
        std::string imagePath = "C:/Progetti/Template C++/example_images/BloodImage_00001.jpeg";

        cv::Mat imgOriginale = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (imgOriginale.empty()) {
            std::cerr << "ERRORE: Impossibile caricare l'immagine." << std::endl;
            return -1;
        }

        std::cout << "Premi un tasto qualsiasi sulla finestra dell'immagine per avanzare allo step successivo..." << std::endl;

        cv::namedWindow("1. Originale", cv::WINDOW_NORMAL);
        cv::imshow("1. Originale", imgOriginale);
        cv::waitKey(0); // PAUSA 1

        // =========================================================================
        // 2. PRE-PROCESSING
        // =========================================================================
        cv::Mat imgColor;
        // Ammorbidiamo l'immagine per ridurre il rumore
        cv::GaussianBlur(imgOriginale, imgColor, cv::Size(5, 5), 0);

        cv::Mat imgHSV, imgGray;
        cv::cvtColor(imgColor, imgHSV, cv::COLOR_BGR2HSV);
        cv::cvtColor(imgColor, imgGray, cv::COLOR_BGR2GRAY);

        // =====================================================================
        // 3. PIPELINE: MASCHERA VIOLA (LEUCOCITI)
        // =====================================================================
        // I "numeri magici" trovati tramite la calibrazione manuale
        cv::Scalar lowerViola(116, 71, 128);
        cv::Scalar upperViola(132, 255, 255);

        cv::Mat maskViola;
        cv::inRange(imgHSV, lowerViola, upperViola, maskViola);

        cv::namedWindow("2. Filtro Colore Diretto (Solo Viola)", cv::WINDOW_NORMAL);
        cv::imshow("2. Filtro Colore Diretto (Solo Viola)", maskViola);
        cv::waitKey(0); // PAUSA 2

        // PULIZIA MORFOLOGICA
        // 1. CLOSING: Tappa i buchi neri all'interno del nucleo (usa un pennello grande)
        cv::morphologyEx(maskViola, maskViola, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));

        // 2. OPENING: Cancella i piccoli puntini bianchi isolati (rumore) sullo sfondo
        cv::morphologyEx(maskViola, maskViola, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

        cv::namedWindow("3. Maschera Viola Pulita", cv::WINDOW_NORMAL);
        cv::imshow("3. Maschera Viola Pulita", maskViola);
        cv::waitKey(0); // PAUSA 3

        // =====================================================================
        // 4. PIPELINE: MASCHERA ROSA (ERITROCITI)
        // =====================================================================
        cv::Mat maskTutteLeCellule;

        // Usiamo Otsu Invertito sull'immagine in Scala di Grigi. 
        // Lo sfondo chiaro diventa nero, le cellule scure diventano bianche.
        cv::threshold(imgGray, maskTutteLeCellule, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        cv::namedWindow("4. Otsu (Tutte le cellule)", cv::WINDOW_NORMAL);
        cv::imshow("4. Otsu (Tutte le cellule)", maskTutteLeCellule);
        cv::waitKey(0); // PAUSA 4

        // Riempiamo le "ciambelle" (i buchi chiari al centro dei globuli rossi)
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(maskTutteLeCellule, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat maskCellulePiene = cv::Mat::zeros(maskTutteLeCellule.size(), CV_8UC1);
        cv::drawContours(maskCellulePiene, contours, -1, cv::Scalar(255), cv::FILLED);
        cv::namedWindow("5. Cellule Piene (Buchi Chiusi)", cv::WINDOW_NORMAL);
        cv::imshow("5. Cellule Piene (Buchi Chiusi)", maskCellulePiene);
        cv::waitKey(0); // PAUSA 5

        // Isoliamo i globuli rossi sottraendo la maschera viola (leucociti) dal totale
        cv::Mat maskRosa;
        cv::subtract(maskCellulePiene, maskViola, maskRosa);

        // Pulizia dei bordi sfrangiati dei globuli rossi
        cv::morphologyEx(maskRosa, maskRosa, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
        cv::namedWindow("6. Maschera Rosa Finale", cv::WINDOW_NORMAL);
        cv::imshow("6. Maschera Rosa Finale", maskRosa);
        cv::waitKey(0); // PAUSA 6
        // =====================================================================
         // 4.5 ALTERNATIVA: HOUGH CIRCLE TRANSFORM (Argomento 20)
         // La tua idea: cercare maschere circolari di dimensione nota!
         // =====================================================================

         // Hough Circles lavora benissimo sull'immagine in scala di grigi originale,
         // ma è sensibile al rumore, quindi la sfochiamo un po'.
        cv::Mat grayBlur;
        cv::medianBlur(imgGray, grayBlur, 5);

        // --- I TUOI PARAMETRI CRITICI DA CALIBRARE ---
        // Quanto è grande un globulo rosso nella tua foto? (in pixel)
        int raggioMinimo = 45;  // Metti il raggio del globulo più piccolo
        int raggioMassimo = 65; // Metti il raggio del globulo più grande

        // Distanza minima tra due centri (se è troppo piccola, trova 2 cerchi sovrapposti sulla stessa cellula)
        int distanzaMinimaTraCentri = raggioMinimo * 0.8;

        // Vettore che conterrà i risultati: [x del centro, y del centro, raggio]
        std::vector<cv::Vec3f> cerchiTrovati;

        // LA MAGIA DI HOUGH
        cv::HoughCircles(grayBlur, cerchiTrovati, cv::HOUGH_GRADIENT, 1,
            distanzaMinimaTraCentri,
            50,  // Parametro 1: Sensibilità ai bordi (Canny). Di solito 50 va bene.
            10,  // Parametro 2: Soglia per i centri. PIÙ È BASSO = Trova più cerchi (anche falsi). PIÙ È ALTO = Trova solo cerchi perfetti.
            raggioMinimo, raggioMassimo);

        // Disegniamo i risultati sull'immagine a colori
        cv::Mat imgRisultatoHough = imgOriginale.clone();

        std::cout << "\nGlobuli Rossi trovati con Hough: " << cerchiTrovati.size() << std::endl;

        for (size_t i = 0; i < cerchiTrovati.size(); i++) {
            cv::Point centro(cvRound(cerchiTrovati[i][0]), cvRound(cerchiTrovati[i][1]));
            int raggio = cvRound(cerchiTrovati[i][2]);

            // Disegna il puntino del centro (verde)
            cv::circle(imgRisultatoHough, centro, 2, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            // Disegna il perimetro del globulo rosso (rosso/fucsia)
            cv::circle(imgRisultatoHough, centro, raggio, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        }

        cv::namedWindow("9. Globuli Rossi (Hough Circles)", cv::WINDOW_NORMAL);
        cv::imshow("9. Globuli Rossi (Hough Circles)", imgRisultatoHough);

        std::cout << "Premi un ultimo tasto per chiudere." << std::endl;
        cv::waitKey(0);
        // =====================================================================
        // 5. ESTRAZIONE FEATURE DA HOUGH CON "DOPPIO CONTROLLO" COLORE
        // =====================================================================
       
        int globuliRossiValidi = 0;

        std::cout << "\n--- ANALISI GLOBULI ROSSI (HOUGH + COLORE) ---" << std::endl;

        for (size_t i = 0; i < cerchiTrovati.size(); i++) {
            int cx = cvRound(cerchiTrovati[i][0]);
            int cy = cvRound(cerchiTrovati[i][1]);
            int raggio = cvRound(cerchiTrovati[i][2]);

            // 0. Sicurezza: Evitiamo che il programma vada in crash se Hough trova un cerchio sul bordo
            if (cx < 0 || cx >= imgOriginale.cols || cy < 0 || cy >= imgOriginale.rows) {
                continue;
            }

            // 1. FILTRO LEUCOCITA: Cade nella maschera viola?
            if (maskViola.at<uchar>(cy, cx) > 0) {
                continue; // È il globulo bianco, ignoralo.
            }

            // 2. IL TUO FILTRO SUPER ROBUSTO: Cade nella maschera rosa?
            // maskRosa è l'immagine dello Step 4/6 (dove i globuli rossi sono bianchi e lo sfondo è nero)
            if (maskRosa.at<uchar>(cy, cx) == 0) {
                // Il centro cade nel NERO. Significa che Hough ha disegnato un cerchio 
                // in mezzo al nulla (falso positivo). Lo scartiamo spietatamente!
                continue;
            }

            // Se sopravvive a tutti i controlli, è un vero globulo rosso!
            globuliRossiValidi++;

            // ESTRAZIONE FEATURE MATEMATICHE (Area del cerchio)
            double area = CV_PI * raggio * raggio;

            std::cout << "Eritrocita ID " << globuliRossiValidi
                << " | Raggio: " << raggio << " px"
                << " | Area Stimata: " << cvRound(area) << " px"
                << " | Centro: (X:" << cx << ", Y:" << cy << ")" << std::endl;

            // Disegna il risultato
            cv::circle(imgRisultatoHough, cv::Point(cx, cy), 2, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            cv::circle(imgRisultatoHough, cv::Point(cx, cy), raggio, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        }

        std::cout << "\nTOTALE GLOBULI ROSSI VALIDI E VERIFICATI: " << globuliRossiValidi << std::endl;

        cv::namedWindow("9. Globuli Rossi (Hough + Colore)", cv::WINDOW_NORMAL);
        cv::imshow("9. Globuli Rossi (Hough + Colore)", imgRisultatoHough);

        std::cout << "Premi un ultimo tasto per chiudere." << std::endl;
        cv::waitKey(0);
    }
    catch (const std::exception& e) {
        std::cerr << "Errore a runtime: " << e.what() << std::endl;
    }
    return 0;
}