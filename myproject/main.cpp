#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem> // Necessario per creare la cartella

namespace fs = std::filesystem;

struct BoundingBox {
    std::string label;
    int x1, y1, x2, y2;
};

// Funzione di lettura JSON
std::vector<BoundingBox> leggiJsonAMano(const std::string& filepath) {
    std::vector<BoundingBox> boxes;
    std::ifstream file(filepath);
    if (!file.is_open()) return boxes;
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t pos = 0;
    while (true) {
        pos = content.find("\"classTitle\": \"", pos);
        if (pos == std::string::npos) break;
        pos += 15;
        size_t endQuote = content.find("\"", pos);
        std::string label = content.substr(pos, endQuote - pos);
        pos = content.find("\"exterior\":", pos);
        if (pos == std::string::npos) break;
        int coords[4];
        for (int i = 0; i < 4; i++) {
            pos = content.find_first_of("0123456789", pos);
            if (pos == std::string::npos) break;
            size_t endNum = content.find_first_not_of("0123456789", pos);
            coords[i] = std::stoi(content.substr(pos, endNum - pos));
            pos = endNum;
        }
        boxes.push_back({ label, coords[0], coords[1], coords[2], coords[3] });
    }
    return boxes;
}

int main() {
    try {
        // 1. PERCORSI
        std::string basePath = "C:\\Progetti\\Template C++\\";
        std::string pathImmagini = basePath + "example_images\\*.jpg";
        std::string folderJson = basePath + "ann\\";

        // Cartella di output visibile in Visual Studio
        std::string folderOutput = basePath + "risultati_elaborati\\";

        // Creazione cartella se non esiste
        if (!fs::exists(folderOutput)) {
            fs::create_directories(folderOutput);
        }

        std::vector<cv::String> filenames;
        cv::glob(pathImmagini, filenames);

        if (filenames.empty()) {
            std::cout << "Nessun file .jpg trovato!" << std::endl;
            return -1;
        }

        std::cout << "Salvataggio in corso in: " << fs::absolute(folderOutput) << "\n" << std::endl;

        for (size_t i = 0; i < filenames.size(); i++) {
            cv::Mat img = cv::imread(filenames[i]);
            if (img.empty()) continue;

            std::string fullPath = filenames[i];
            size_t lastSlash = fullPath.find_last_of("/\\");
            std::string fileNameOnly = (lastSlash == std::string::npos) ? fullPath : fullPath.substr(lastSlash + 1);
            size_t lastDot = fileNameOnly.find_last_of(".");
            std::string baseName = fileNameOnly.substr(0, lastDot);

            std::string jsonPath = folderJson + baseName + ".jpeg.json";
            std::vector<BoundingBox> oggettiTrovati = leggiJsonAMano(jsonPath);

            // Disegno dei box
            for (const auto& obj : oggettiTrovati) {
                cv::Scalar color;
                if (obj.label == "WBC") {
                    color = cv::Scalar(0, 255, 0);   // VERDE per i Bianchi
                }
                else if (obj.label == "RBC") {
                    color = cv::Scalar(0, 0, 255);   // ROSSO per i Rossi
                }
                else if (obj.label == "Platelets") {
                    color = cv::Scalar(255, 255, 0); // CIANO/GIALLO per le Piastrine
                }
                else {
                    color = cv::Scalar(255, 255, 255); // BIANCO per altro
                }
                cv::rectangle(img, cv::Point(obj.x1, obj.y1), cv::Point(obj.x2, obj.y2), color, 2);
                cv::putText(img, obj.label, cv::Point(obj.x1, obj.y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            }

            // 2. SALVATAGGIO FISICO
            std::string savePath = folderOutput + baseName + "_elaborata.jpg";
            cv::imwrite(savePath, img);

            // Stampa del percorso per ogni file
            std::cout << "[" << i + 1 << "/" << filenames.size() << "] Salvato: " << savePath << std::endl;
        }

        std::cout << "\n--- OPERAZIONE COMPLETATA ---" << std::endl;
        std::cout << "Tutte le immagini sono pronte nella cartella: " << folderOutput << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
    }
    return 0;
}