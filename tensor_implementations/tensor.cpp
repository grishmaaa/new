#pragma once

#include <iostream>
#include <sstream>
#include <random>
#include <cmath>
#include <iomanip>

template<typename T>
class tensor {
public:
    const unsigned int x, y, z, s;

    tensor(unsigned int x, unsigned int y, unsigned int z, T val) : x(x), y(y), z(z), s(x * y * z) {
        p_data = new T[s];
        for (unsigned int i = 0; i < s; i++) p_data[i] = val;
    }

    tensor(const tensor<T> & other) : x(other.x), y(other.y), z(other.z), s(other.s) {
        p_data = new T[s];
        memcpy(p_data, other.get_data(), s * sizeof(T));
    }

    ~tensor() {
        delete[] p_data;
        p_data = nullptr;
    }

    T * get_data() {
        return p_data;
    }

    static tensor<T> * random(unsigned int x, unsigned int y, unsigned int z, T val, T min, T max) {
        tensor<T> * p_tensor = new tensor<T>(x, y, z, val);

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<T> dist(min, max);

        for (unsigned int i = 0; i < p_tensor->s; i++) {
            T rnd = dist(mt);
            while (abs(rnd) < 0.001) rnd = dist(mt);
            p_tensor->get_data()[i] = rnd;
        }

        return p_tensor;
    }

    static tensor<T> * from(std::vector<T> * p_data, T val) {
        tensor<T> * p_tensor = new tensor<T>(p_data->size(), 1, 1, val);

        for (unsigned int i = 0; i < p_tensor->get_x(); i++) p_tensor->set_data(i + 0 * p_tensor->get_x() * + 0 * p_tensor->get_x() * p_tensor->get_y(), p_data->at(i));

        return p_tensor;
    }

    friend std::ostream & operator <<(std::ostream & stream, tensor<T> & tensor) {
        stream << "(" << tensor.x << "," << tensor.y << "," << tensor.z << ") Tensor\n";

        for (unsigned int i = 0; i < tensor.x; i++) {
            for (unsigned int k = 0; k < tensor.z; k++) {
                stream << "[";

                for (unsigned int j = 0; j < tensor.y; j++) {
                    stream << std::setw(5) << roundf(tensor(i, j, k) * 1000) / 1000;
                    if (j + 1 < tensor.y) stream << ",";
                }

                stream << "]";

            }

            stream << std::endl;
        }

        return stream;
    }

    tensor<T> & operator +(tensor<T> & other) {
        tensor<T> result(*this);

        return result;
    }

    tensor<T> & operator -(tensor<T> & other) {
        tensor<T> result(*this);

        return result;
    }

    tensor<T> & operator *(tensor<T> & other) {
        tensor<T> result(*this);

        return result;
    }

    T & operator ()(unsigned int i, unsigned int j, unsigned int k) {
        return p_data[i + (j * x) + (k * x * y)];
    }

    T & operator ()(unsigned int i) {
        return p_data[i];
    }

private:
    T * p_data = nullptr;
};

int main() {
    tensor<double> * p_tensor_input = tensor<double>::random(6, 2, 3, 0.0, 0.0, 1.0);
    tensor<double> * p_tensor_weight = tensor<double>::random(2, 6, 3, 0.0, 0.0, 1.0);

    std::cout << *p_tensor_input << std::endl;
    std::cout << *p_tensor_weight << std::endl;

    tensor<double> p_tensor_output = *p_tensor_input + *p_tensor_weight;

    return 0;
}