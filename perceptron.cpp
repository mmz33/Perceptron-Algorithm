#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <numeric>
#include <algorithm>

/**
 * Perceptron Algorithm
 * @author Mohammad Zeineldeen
 */

class Point {
private:
    std::vector<float> data; // training data points
    int label; // +1 or -1
public:
    Point(std::vector<float> coordinates, int l) :
      data(coordinates), label(l) {
        data.push_back(1.f); // extend space for fast convergence
    }

    // Returns the norm of this point
    float get_norm() const {
        float sum = std::accumulate(data.begin(), data.end(), 0.f,
                                [](float& res, const float& x) {
                                  return res += x * x;
                                });
        return std::sqrt(sum);
    }

    // Normalizes this point
    void normalize_point(const float max_norm) {
        std::transform(data.begin(),
                       data.end(),
                       data.begin(),
                       std::bind2nd(std::divides<float>(), max_norm));
    }

    int dims()      const { return data.size(); }
    int get_label() const { return label; }

    const float& operator[](int idx) const {
      return data[idx];
    }

    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
      os << "([";
      for (int i = 0; i < p.dims(); ++i) {
        if (i > 0) os << ", ";
        os << p[i];
      }
      os << "], " << p.label << ")";
      return os;
    }
};

class Perceptron {
private:
    std::vector<Point> points; // vector of training data points
    std::vector<float> w; // weight vector
    int dim;
public:
    Perceptron(std::vector<Point> pnts) : points(pnts) {
        dim = points[0].dims();
        w = std::vector<float>(dim);
    }

    // Checks if the sum is positive or negative
    int result(float sum) {
        return std::signbit(sum) ? -1 : +1;
    }

    // Returns the sign of the result (-1 or +1)
    int sgn(const Point& p) {
        float sum = 0.f;
        for (int i = 0; i < dim; ++i) {
            sum += w[i] * p[i];
        }
        return result(sum);
    }

    // update the weight vector
    void update_weight(const Point& p) {
        for (int i = 0; i < dim; ++i) {
            w[i] += p.get_label() * p[i];
        }
    }

    // print data set
    void print_data_set() const {
        std::cout << "Data Set (Normalized):\n";
        std::cout << "----------------------\n";
        for (const auto& p: points) {
          std::cout << p << std::endl;
        }
        std::cout << std::endl;
    }

    // Prints the weight vector
    void print_weight() const {
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
                if (sgn(point) * point.get_label() < 0) {
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

    float max_norm = std::numeric_limits<float>::min();
    std::vector<Point> points;

    // Read input
    for (int i = 0; i < num_of_pnts; ++i) {
        std::vector<float> data(dim);
        for (int j = 0; j < dim; ++j) in >> data[j];
        int label; in >> label;
        Point p {data, label};
        points.push_back(p);
        max_norm = std::max(max_norm, p.get_norm());
    }

    for (auto& point: points) {
        point.normalize_point(max_norm);
    }

    Perceptron perc {points};
    perc.print_data_set();
    perc.run(1000);
    perc.print_final_res(max_norm);
    return 0;
}
