#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    try {
        // =========================================================================
        // 1. SETUP PERCORSI PARALLELI (ORIGINALI VS ANNOTATE)
        // =========================================================================
        std::string folderOriginali = "C:/Progetti/Template C++/example_images/";
        std::string folderAnnotate  = "C:/Progetti/Template C++/output/";
        
        std::string outFolderBianchi = "C:/Progetti/Template C++/output_bianchi/";
        std::string outFolderPiastrine = "C:/Progetti/Template C++/output_piastrine/";

        fs::create_directories(outFolderBianchi);
        fs::create_directories(outFolderPiastrine);

        // Usiamo la cartella delle originali per impostare il ciclo di lettura
        std::vector<cv::String> imagePaths;
        cv::glob(folderOriginali + "*.jpeg", imagePaths);

        if (imagePaths.empty()) {
            std::cerr << "ERRORE: Nessuna immagine originale trovata in example_images." << std::endl;
            return -1;
        } 

        // I tuoi parametri corretti dal calibratore Python (H, S, V)
        cv::Scalar lowerViolaGlobale(78, 23, 161);
        cv::Scalar upperViolaGlobale(134, 255, 252);

        // =========================================================================
        // 2. CICLO DI ELABORAZIONE
        // =========================================================================
        for (size_t f = 0; f < imagePaths.size(); f++) {
            // A. Carichiamo l'immagine ORIGINALE su cui faremo TUTTI i calcoli
            cv::Mat imgOriginale = cv::imread(imagePaths[f], cv::IMREAD_COLOR);
            if (imgOriginale.empty()) continue;

            // Estraiamo il nome del file (es. BloodImage_00005.jpeg)
            std::string fullPath = imagePaths[f];
            size_t lastSlash = fullPath.find_last_of("/\\");
            std::string fileName = fullPath.substr(lastSlash + 1);

            // B. Carichiamo l'immagine ANNOTATA corrispondente (solo per la visualizzazione)
            std::string pathAnnotata = folderAnnotate + fileName;
            cv::Mat imgAnnotataReale = cv::imread(pathAnnotata, cv::IMREAD_COLOR);
            
            if (imgAnnotataReale.empty()) {
                std::cout << "Avviso: Manca l'immagine annotata per " << fileName << ". Mostrero' solo l'originale." << std::endl;
                imgAnnotataReale = imgOriginale.clone(); // Fallback se manca il file annotato
            }

            std::cout << "\n--- Elaborazione in corso: " << fileName << " (" << f + 1 << "/" << imagePaths.size() << ") ---" << std::endl;

            // =====================================================================
            // FASE A: PRE-PROCESSING (PULIZIA RIGIDA SULL'IMMAGINE ORIGINALE)
            // =====================================================================
            cv::Mat imgMedian, imgBilateral;
            cv::medianBlur(imgOriginale, imgMedian, 3);
            cv::bilateralFilter(imgMedian, imgBilateral, 9, 75, 75);

            cv::Mat imgHSV;
            cv::cvtColor(imgBilateral, imgHSV, cv::COLOR_BGR2HSV);

            // =====================================================================
            // FASE B: MASCHERA COLORE (DALL'ORIGINALE PULITA)
            // =====================================================================
            cv::Mat maskGlobale;
            cv::inRange(imgHSV, lowerViolaGlobale, upperViolaGlobale, maskGlobale);

            // Morfologia matematica per compattare i nuclei e uccidere la polvere
            cv::morphologyEx(maskGlobale, maskGlobale, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
            cv::morphologyEx(maskGlobale, maskGlobale, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

            // =====================================================================
            // FASE C: SMISTAMENTO PER AREA
            // =====================================================================
            cv::Mat maskSoloBianchi = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
            cv::Mat maskSoloPiastrine = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);

            cv::Mat labels, stats, centroids;
            int nLabels = cv::connectedComponentsWithStats(maskGlobale, labels, stats, centroids);

            for (int i = 1; i < nLabels; i++) {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);

                if (area >= 800) {
                    maskSoloBianchi.setTo(255, labels == i); 
                }
                else if (area >= 10 && area <= 300) {
                    maskSoloPiastrine.setTo(255, labels == i); 
                }
            }

            // Dilatazione estetica delle piastrine per renderle ben visibili a schermo
            cv::dilate(maskSoloPiastrine, maskSoloPiastrine, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

            // =====================================================================
            // FASE D: SALVATAGGIO MASCHERE PULITE
            // =====================================================================
            cv::imwrite(outFolderBianchi + fileName, maskSoloBianchi);
            cv::imwrite(outFolderPiastrine + fileName, maskSoloPiastrine);

            // =====================================================================
            // FASE E: FINESTRE DI CONTROLLO (CONFRONTO CORRETTO)
            // =====================================================================
            // Finestra 1: L'immagine originale pulita del dataset (su cui hai lavorato)
            cv::namedWindow("1. Immagine Originale di Lavoro", cv::WINDOW_NORMAL);
            cv::imshow("1. Immagine Originale di Lavoro", imgOriginale);

            // Finestra 2: L'immagine annotata con i rettangoli (usata come guida visiva per te)
            cv::namedWindow("2. GUIDA REALE (Con Bounding Box)", cv::WINDOW_NORMAL);
            cv::imshow("2. GUIDA REALE (Con Bounding Box)", imgAnnotataReale);

            // Finestra 3: La maschera dei Bianchi estratta dall'originale
            cv::namedWindow("3. TUA MASK - GLOBULI BIANCHI", cv::WINDOW_NORMAL);
            cv::imshow("3. TUA MASK - GLOBULI BIANCHI", maskSoloBianchi);

            // Finestra 4: La maschera delle Piastrine estratta dall'originale
            cv::namedWindow("4. TUA MASK - PIASTRINE", cv::WINDOW_NORMAL);
            cv::imshow("4. TUA MASK - PIASTRINE", maskSoloPiastrine);

            // Premi un tasto qualsiasi sulle finestre per avanzare nel dataset, ESC per uscire
            int key = cv::waitKey(0);
            if (key == 27) {
                std::cout << "\nInterruzione manuale." << std::endl;
                break;
            }
        }

        std::cout << "\n[FINE] Controllo incrociato completato!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
    }
    return 0;
}