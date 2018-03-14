#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <fstream>
#include <iomanip>

/**
 * Perceptron Algorithm
 * @author Mohammad_ZD
 */

class Point {
public:
    std::vector<long double> data; // training data points
    int output; // +1 or -1

    Point(std::vector<long double> coordinates, int res) :
        data(coordinates), output(res) {

        data.push_back(1.0); // extend space for fast convergence
    }

    // Returns the norm of this point
    long double get_norm() {
        long double sum = 0.0;
        for (const auto& x: data) {
            sum += x*x;
        }
        return std::sqrt(sum);
    }

    // Normalizes this point
    void normalize_point(long double max_norm) {
        for (auto& coor: data) {
            coor = coor/max_norm;
        }
    }
};

class Perceptron {
public:
    std::vector<Point> points; // vector of training data points
    std::vector<long double> w; // weight vector
    int dim;

    Perceptron(std::vector<Point> pnts) : points(pnts) {
        dim = pnts[0].data.size();
        w = std::vector<long double>(dim, 0.0);
    }

    // Checks if the sum is positive or negative
    int result(long double sum) {
        return std::signbit(sum) ? -1 : +1;
    }

    // Returns the sign of the result (-1 or +1)
    int sgn(Point p) {
        long double sum = 0.0;
        for (int i = 0; i < dim; ++i) {
            sum += w[i] * p.data[i];
        }
        return result(sum);
    }

    // update the weight vector
    void update_weight(Point p) {
        for (int i = 0; i < dim; ++i) {
            w[i] += p.output * p.data[i];
        }
    }

    // print data set
    void print_data_set() {
        std::cout << "Data Set (Normalized):\n";
        std::cout << "----------------------\n";
        for (const auto& p: points) {
            std::cout << "[";
            for (int i = 0; i < dim; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << p.data[i];
            }
            std::cout << "]\n";
        }
        std::cout << '\n';
    }

    // Prints the weight vector
    void print_weight() {
        std::cout << "[";
        for (int i = 0; i < dim; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << w[i];
        }
        std::cout << "]\n";
    }

    // Runs the Perceptron Algorithm
    void run(int max_num_of_iterations) {
        int iterations = 0, max_iterations = max_num_of_iterations;
        std::cout << "Start iterations:\n";
        std::cout << "-----------------\n";
        while (iterations <= max_iterations) {
            bool change = false;
            for (const auto& point: points) {
                if (sgn(point) != point.output) {
                    update_weight(point);
                    std::cout << "Iteration " << iterations << ": ";
                    print_weight();
                    change = true;
                }
            }
            if (!change) break;
            iterations++;
        }
    }

    // Print final weight vector
    void print_final_res(long double max_norm) {
        std::cout << "\nFinal result:\n";
        std::cout << "-------------\n";
        std::cout << "Before normalization: W = ";
        print_weight();
        std::cout << "After normalization:  W = [";
        for (int i = 0; i < dim; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << w[i] * max_norm;
        }
        std::cout << "]\n";
    }
};

int main() {
    std::ifstream in("input.in");

    int num_of_pnts, dim; in >> num_of_pnts >> dim;

    long double max_norm = -1;
    std::vector<Point> points;

    // Read input
    for (int i = 0; i < num_of_pnts; ++i) {
        std::vector<long double> data(dim, 0.0);
        for (int j = 0; j < dim; ++j) {
            in >> data[j];
        }
        int label; in >> label;
        Point p = {data, label};
        points.push_back(p);
        max_norm = std::max(max_norm, p.get_norm());
    }

    for (auto& point: points) {
        point.normalize_point(max_norm);
    }

    Perceptron perc = {points};

    perc.print_data_set();
    perc.run(1000);
    perc.print_final_res(max_norm);

    return 0;
}
