#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <vector>
 
using Eigen::MatrixXd;
using features_t = std::vector<float>;
 
Eigen::MatrixXf read_mat_from_stream(size_t rows, size_t cols, std::istream& stream) {
    Eigen::MatrixXf res(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float val;
            stream >> val;
            res(i, j) = val;
        }
    }
    return res;
}

bool read_features(std::istream& stream, features_t& features) {
    std::string line;
    std::getline(stream, line);

    features.clear();
    features.push_back(1.0f);
    std::istringstream linestream{line};
    double value;
    while (linestream >> value) {
        features.push_back(value);
    }
    return stream.good();
}

bool read_features_csv(std::istream& stream, features_t& features) {
    std::string line;
    std::getline(stream, line);
    features.clear();
    features.push_back(1.0f);
    std::istringstream linestream{line};
    double value;
    char delimiter = ',';
    if (linestream.peek() == delimiter) {
            linestream.ignore();
        }
    while (linestream >> value) {
        features.push_back(value);
        if (linestream.peek() == delimiter) {
            linestream.ignore();
        }
    }
    return stream.good();
}

Eigen::MatrixXf read_mat_from_file(size_t rows, size_t cols, const std::string& filepath) {
    std::ifstream stream{filepath};
    return read_mat_from_stream(rows, cols, stream);
}

size_t predict(Eigen::MatrixXf& w, features_t& f) {
    Eigen::VectorXf x = Eigen::VectorXf::Map(&f[0], f.size());
    //std::cout << x << std::endl;
    auto res = w * x;
    int pos = 0;
    res.maxCoeff(&pos);
    return pos;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "Usage: " << " <path_to_data> " << "<path_to_model>" << std::endl;
        return 1;
    }
    auto w = read_mat_from_file(10, 785, argv[2]); // "logreg_coef.txt"
    auto features = features_t{};

    std::ifstream test_data_csv{argv[1]}; //{"test.csv"};
    int count = 0;
    int correct_count = 0;
    for (;; count++) {
        size_t y_true;
        test_data_csv >> y_true;
        if (!read_features_csv(test_data_csv, features)) {
            break;
        }
        auto y_pred = predict(w, features);
        if (y_pred == y_true) {
            correct_count++;
        }
    }
    std::cout << static_cast<float>(correct_count)/count << std::endl;
}
