#include "ipaConfig.h"
#include "ucasConfig.h"
#include <fstream>
#include <vector>
#include "functions.h"

int main()
{
    try
    {
        // 1. LETTURA AUTOMATICA DELLA CARTELLA
        std::string folderPath = std::string(EXAMPLE_IMAGES_PATH);
        std::vector<cv::String> filenames;

        // Cerca tutti i file .jpeg nella cartella
        cv::glob(folderPath + "/*.jpeg", filenames);

        if (filenames.empty())
            throw ipa::error("Nessuna immagine trovata nella cartella specificata");

        printf("Trovate %zu immagini. Inizio estrazione...\n", filenames.size());

        // 2. CREAZIONE CSV CON COLONNA "IMAGE_NAME"
        std::ofstream csvFile("dataset_estratto.csv");
        csvFile << "Image_Name;ID;Centroid_X;Centroid_Y;Area;Width;Height;Compactness;Eccentricity\n";

        // 3. CICLO SU TUTTE LE IMMAGINI
        for (size_t f = 0; f < filenames.size(); f++)
        {
            cv::Mat imgGray8 = cv::imread(filenames[f], cv::IMREAD_GRAYSCALE);
            if (!imgGray8.data) continue;

            // Estrazione del nome del file per il CSV
            std::string fullPath = filenames[f];
            std::string fileName = fullPath.substr(fullPath.find_last_of("/\\") + 1);

            //printf("Elaborazione: %s\n", fileName.c_str());

            // --- PROCESSING ---
            cv::Mat imgFloat;
            imgGray8.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

            //trasformazione logaritmica
            cv::log(imgFloat + 1.0f, imgFloat);
            cv::normalize(imgFloat, imgGray8, 0, 255, cv::NORM_MINMAX, CV_8U);

            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
            cv::morphologyEx(imgGray8, imgGray8, cv::MORPH_TOPHAT, kernel);

            cv::Mat imgBin;
            cv::threshold(imgGray8, imgBin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

            // --- Aggiungi questo blocco DOPO la binarizzazione di Otsu e PRIMA dell'estrazione delle feature ---

// --- Aggiungi questo blocco DOPO la binarizzazione di Otsu e PRIMA dell'estrazione delle feature ---

           // --- INIZIO BLOCCO DI DEBUG (Solo per la prima immagine) ---
            if (f == 0)
            {
                // 1. Immagine originale (ricaricata per confronto)
                cv::Mat viewOrig = cv::imread(filenames[f], cv::IMREAD_GRAYSCALE);
                ipa::imshow("1. Originale (Grayscale)", viewOrig);

                // 2. Post-Logaritmo (Passaggio intermedio "puro")
                cv::Mat tempFloat, viewLog;
                viewOrig.convertTo(tempFloat, CV_32F, 1.0 / 255.0);
                cv::log(tempFloat + 1.0f, tempFloat);
                cv::normalize(tempFloat, viewLog, 0, 255, cv::NORM_MINMAX, CV_8U);
                ipa::imshow("2. Post-Logaritmo", viewLog);

                // 3. Post-White Top-Hat
                cv::Mat viewTopHat;
                cv::Mat kernelVisual = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
                cv::morphologyEx(viewLog, viewTopHat, cv::MORPH_TOPHAT, kernelVisual);
                ipa::imshow("3. Post-White Top-Hat", viewTopHat);

                // 4. Risultato Binarizzazione di Otsu
                ipa::imshow("4. Binarizzazione (Otsu)", imgBin);

                // 5. Risultato Finale (Pulizia con Apertura)
                cv::Mat viewFinal;
                cv::Mat kernelSmall = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
                cv::morphologyEx(imgBin, viewFinal, cv::MORPH_OPEN, kernelSmall);
                ipa::imshow("5. Risultato Pulito (Ready for Features)", viewFinal);

                // 6. Connected Components (Colori Falsi)
                cv::Mat viewLabels, viewStats, viewCentroids;
                int nObjectsForView = cv::connectedComponentsWithStats(imgBin, viewLabels, viewStats, viewCentroids);

                if (nObjectsForView > 1)
                {
                    cv::Mat viewConnectedColor = cv::Mat::zeros(imgBin.size(), CV_8UC3);
                    std::vector<cv::Vec3b> colors(nObjectsForView);
                    colors[0] = cv::Vec3b(0, 0, 0); // Sfondo nero

                    for (int i = 1; i < nObjectsForView; i++)
                        colors[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);

                    for (int r = 0; r < viewConnectedColor.rows; r++) {
                        for (int c = 0; c < viewConnectedColor.cols; c++) {
                            int label = viewLabels.at<int>(r, c);
                            viewConnectedColor.at<cv::Vec3b>(r, c) = colors[label];
                        }
                    }
                    ipa::imshow("6. Connected Components (Colori Falsi)", viewConnectedColor);
                }
                else
                {
                    printf("Nessun oggetto trovato per la visualizzazione Components.\n");
                }

                printf("Anteprima mostrata per la prima immagine. Premi un tasto su una finestra per continuare il ciclo...\n");
                cv::waitKey(0); // Ferma l'esecuzione solo qui
            }
            // --- FINE BLOCCO DI DEBUG ---
            // 
            // --- ESTRAZIONE FEATURE ---
            cv::Mat labels, stats, centroids;
            int nObjects = cv::connectedComponentsWithStats(imgBin, labels, stats, centroids);

            for (int i = 1; i < nObjects; i++) {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);

                // FILTRO RUMORE (abbassato a 3 per catturare anche le piastrine)
                if (area > 3) {
                    double cx = centroids.at<double>(i, 0);
                    double cy = centroids.at<double>(i, 1);
                    int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
                    int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

                    double compactness = (double)area / (width * height);

                    int x = stats.at<int>(i, cv::CC_STAT_LEFT);
                    int y = stats.at<int>(i, cv::CC_STAT_TOP);
                    cv::Rect roi(x, y, width, height);

                    cv::Mat objMask = (labels(roi) == i);
                    cv::Moments m = cv::moments(objMask, true);

                    double eccentricity = 0.0;
                    if (m.m00 > 0) {
                        double delta = std::sqrt(4 * m.mu11 * m.mu11 + (m.mu20 - m.mu02) * (m.mu20 - m.mu02));
                        double lambda1 = (m.mu20 + m.mu02 + delta) / 2.0;
                        double lambda2 = (m.mu20 + m.mu02 - delta) / 2.0;

                        if (lambda1 > 0) {
                            eccentricity = std::sqrt(1.0 - (lambda2 / lambda1));
                        }
                    }

                    // Scrittura nel CSV includendo il nome del file all'inizio
                    csvFile << fileName << ";" << i << ";" << cx << ";" << cy << ";"
                        << area << ";" << width << ";" << height << ";"
                        << compactness << ";" << eccentricity << "\n";
                }
            }
        }

        csvFile.close();
        printf("Estrazione completata! File salvato come dataset_estratto.csv\n");
        system("start dataset_estratto.csv");

        return EXIT_SUCCESS;
    }
    catch (ipa::error& ex) {
        std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
    }
    catch (cv::Exception& ex) {
        std::cout << "OPENCV EXCEPTION:\n\t|=> " << ex.what() << std::endl;
    }
}