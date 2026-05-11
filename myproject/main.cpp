#include "ipaConfig.h"
#include "ucasConfig.h"
#include <fstream>
#include "functions.h"


int main()

{
   
    try
         skdjskjdhksjdfhksjh
    {
        
        //grayscale image (8 bits per pixel)

        cv::Mat imgGray8 = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/gal.jpg", cv::IMREAD_GRAYSCALE);

        if (!imgGray8.data)

            throw ipa::error("Cannot load image");

        printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n",

            imgGray8.rows, imgGray8.cols, imgGray8.channels(), ipa::bitdepth(imgGray8.depth()));

        ipa::imshow("An 8-bit grayscale image", imgGray8);



        //  Conversione in float (in questo modo i pixel hanno valori decimali più precisi invece che da 0 a 255)

        cv::Mat imgFloat;

        imgGray8.convertTo(imgFloat, CV_32F, 1.0 / 255.0);



        //  Applicazione del logaritmo per esaltare le sorgenti deboli

        cv::log(imgFloat + 1.0f, imgFloat);



        //  Normalizzazione e ritorno al formato 8-bit per la visualizzazione

        cv::normalize(imgFloat, imgGray8, 0, 255, cv::NORM_MINMAX, CV_8U);



        ipa::imshow("Immagine Post-Logaritmo", imgGray8);



        //  Creazione dell'elemento strutturante (un cerchio di raggio 15)

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));



        //  Applicazione del filtro Top-Hat (White Top-Hat)

        cv::morphologyEx(imgGray8, imgGray8, cv::MORPH_TOPHAT, kernel);


       ipa::imshow("Immagine Post-TopHat", imgGray8);



        //  Creazione della maschera binaria con l'algoritmo di Otsu

        cv::Mat imgBin;

        cv::threshold(imgGray8, imgBin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);


        ipa::imshow("Maschera Binaria", imgBin);



        // Estrazione delle componenti connesse (oggetti isolati)

        cv::Mat labels, stats, centroids;

        int nObjects = cv::connectedComponentsWithStats(imgBin, labels, stats, centroids);



        // Stampa il numero di oggetti trovati (escludendo lo sfondo)

        printf("Numero di oggetti rilevati: %d\n", nObjects - 1);



        // Creazione e apertura del file CSV

        std::ofstream csvFile("dataset_estratto.csv");



        // intestazione 

        csvFile << "ID;Centroid_X;Centroid_Y;Area;Width;Height;Compactness;Eccentricity";


        // Ciclo for per analizzare ogni singolo oggetto

        for (int i = 1; i < nObjects; i++) {


            int area = stats.at<int>(i, cv::CC_STAT_AREA);


            // FILTRO RUMORE: Analizziamo solo gli oggetti con area maggiore di 10 pixel

            if (area > 10) {

                // Estrazione coordinate del centro

                double cx = centroids.at<double>(i, 0);

                double cy = centroids.at<double>(i, 1);



                // Estrazione dimensioni bounding box

                int width = stats.at<int>(i, cv::CC_STAT_WIDTH);

                int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);



                // CALCOLO COMPATTEZZA: Area / (Larghezza * Altezza)

               double compactness = (double)area / (width * height);

               // CALCOLO ECCENTRICITÀ TRAMITE MOMENTI
               
                // Estraiamo le coordinate del bounding box per creare una ROI (Region of Interest)

               int x = stats.at<int>(i, cv::CC_STAT_LEFT);
               int y = stats.at<int>(i, cv::CC_STAT_TOP);
               cv::Rect roi(x, y, width, height);

               // Creiamo una maschera binaria solo per l'oggetto corrente (molto più veloce che farla su tutta l'immagine)

               cv::Mat objMask = (labels(roi) == i);

               // Calcoliamo i momenti

               cv::Moments m = cv::moments(objMask, true);

               double eccentricity = 0.0;

               // Evitiamo divisioni per zero assicurandoci che l'oggetto abbia "massa"

               if (m.m00 > 0) {
                   double mu20 = m.mu20;
                   double mu02 = m.mu02;
                   double mu11 = m.mu11;

                   // Calcolo degli autovalori

                   double delta = std::sqrt(4 * mu11 * mu11 + (mu20 - mu02) * (mu20 - mu02));
                   double lambda1 = (mu20 + mu02 + delta) / 2.0; // asse maggiore
                   double lambda2 = (mu20 + mu02 - delta) / 2.0; // asse minore

                   // Se l'asse maggiore è valido, calcoliamo l'eccentricità

                   if (lambda1 > 0) {
                       eccentricity = std::sqrt(1.0 - (lambda2 / lambda1));
                   }

				   // Scriviamo i dati estratti nel file CSV
                   csvFile << i << ";" << cx << ";" << cy << ";" << area << ";" << width << ";" << height << ";" << compactness << ";" << eccentricity << "\n";
               }
            }

        }



        csvFile.close(); // Chiudiamo il file per salvare i dati

        printf("Estrazione completata con successo! Dati salvati in dataset_estratto.csv\n");


        // Comando per aprire automaticamente il file CSV (specifico per Windows)

        system("start dataset_estratto.csv");





        return EXIT_SUCCESS;

    }

    catch (ipa::error& ex)

    {

        std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;

    }

    catch (ucas::Error& ex)

    {

        std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;

    }
    catch (cv::Exception& ex)
    {
        std::cout << "OPENCV EXCEPTION:\n\t|=> " << ex.what() << std::endl;
    }

}