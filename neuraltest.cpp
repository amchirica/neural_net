{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO7FNuBGGSN0GURNV+XV0Aa",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/reira08/reira08/blob/circleci-project-setup/neuraltest.cpp\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "avFQnqEZ0J4y",
        "outputId": "0abd9b40-af47-4bc9-c371-a0215d8369bb"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing NeuralNetwork.cpp\n"
          ]
        }
      ],
      "source": [
        "%%writefile NeuralNetwork.cpp\n",
        "// NeuralNetwork.hpp\n",
        "#include <iostream>\n",
        "#include <vector>\n",
        " \n",
        "// use typedefs for future ease for changing data types like : float to double\n",
        "typedef float Scalar;\n",
        "typedef Eigen::MatrixXf Matrix;\n",
        "typedef Eigen::RowVectorXf RowVector;\n",
        "typedef Eigen::VectorXf ColVector;\n",
        " \n",
        "// neural network implementation class!\n",
        "class NeuralNetwork {\n",
        "public:\n",
        "    // constructor\n",
        "    NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));\n",
        " \n",
        "    // function for forward propagation of data\n",
        "    void propagateForward(RowVector& input);\n",
        " \n",
        "    // function for backward propagation of errors made by neurons\n",
        "    void propagateBackward(RowVector& output);\n",
        " \n",
        "    // function to calculate errors made by neurons in each layer\n",
        "    void calcErrors(RowVector& output);\n",
        " \n",
        "    // function to update the weights of connections\n",
        "    void updateWeights();\n",
        " \n",
        "    // function to train the neural network give an array of data points\n",
        "    void train(std::vector<RowVector*> data);\n",
        " \n",
        "    // storage objects for working of neural network\n",
        "    /*\n",
        "          use pointers when using std::vector<Class> as std::vector<Class> calls destructor of\n",
        "          Class as soon as it is pushed back! when we use pointers it can't do that, besides\n",
        "          it also makes our neural network class less heavy!! It would be nice if you can use\n",
        "          smart pointers instead of usual ones like this\n",
        "        */\n",
        "    std::vector<RowVector*> neuronLayers; // stores the different layers of out network\n",
        "    std::vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers\n",
        "    std::vector<RowVector*> deltas; // stores the error contribution of each neurons\n",
        "    std::vector<Matrix*> weights; // the connection weights itself\n",
        "    Scalar learningRate;\n",
        "};"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "! g++ NeuralNetwork.cpp -o neuraltest"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2uUfYm_P1bO3",
        "outputId": "7714f558-1c14-4c67-d7b4-db31e79565e9"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[01m\u001b[KNeuralNetwork.cpp:7:9:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KEigen\u001b[m\u001b[K’ does not name a type\n",
            " typedef \u001b[01;31m\u001b[KEigen\u001b[m\u001b[K::MatrixXf Matrix;\n",
            "         \u001b[01;31m\u001b[K^~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:8:9:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KEigen\u001b[m\u001b[K’ does not name a type\n",
            " typedef \u001b[01;31m\u001b[KEigen\u001b[m\u001b[K::RowVectorXf RowVector;\n",
            "         \u001b[01;31m\u001b[K^~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:9:9:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KEigen\u001b[m\u001b[K’ does not name a type\n",
            " typedef \u001b[01;31m\u001b[KEigen\u001b[m\u001b[K::VectorXf ColVector;\n",
            "         \u001b[01;31m\u001b[K^~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:18:27:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KRowVector\u001b[m\u001b[K’ has not been declared\n",
            "     void propagateForward(\u001b[01;31m\u001b[KRowVector\u001b[m\u001b[K& input);\n",
            "                           \u001b[01;31m\u001b[K^~~~~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:21:28:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KRowVector\u001b[m\u001b[K’ has not been declared\n",
            "     void propagateBackward(\u001b[01;31m\u001b[KRowVector\u001b[m\u001b[K& output);\n",
            "                            \u001b[01;31m\u001b[K^~~~~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:24:21:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KRowVector\u001b[m\u001b[K’ has not been declared\n",
            "     void calcErrors(\u001b[01;31m\u001b[KRowVector\u001b[m\u001b[K& output);\n",
            "                     \u001b[01;31m\u001b[K^~~~~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:28:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KRowVector\u001b[m\u001b[K’ was not declared in this scope\n",
            "     void train(std::vector<\u001b[01;31m\u001b[KRowVector\u001b[m\u001b[K*> data);\n",
            "                            \u001b[01;31m\u001b[K^~~~~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:38:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 1 is invalid\n",
            "     void train(std::vector<RowVector*\u001b[01;31m\u001b[K>\u001b[m\u001b[K data);\n",
            "                                      \u001b[01;31m\u001b[K^\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:38:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 2 is invalid\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:28:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KRowVector\u001b[m\u001b[K’ was not declared in this scope\n",
            "     void train(std::vector<\u001b[01;31m\u001b[KRowVector\u001b[m\u001b[K*> data);\n",
            "                            \u001b[01;31m\u001b[K^~~~~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:38:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 1 is invalid\n",
            "     void train(std::vector<RowVector*\u001b[01;31m\u001b[K>\u001b[m\u001b[K data);\n",
            "                                      \u001b[01;31m\u001b[K^\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:38:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 2 is invalid\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:28:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KRowVector\u001b[m\u001b[K’ was not declared in this scope\n",
            "     void train(std::vector<\u001b[01;31m\u001b[KRowVector\u001b[m\u001b[K*> data);\n",
            "                            \u001b[01;31m\u001b[K^~~~~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:38:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 1 is invalid\n",
            "     void train(std::vector<RowVector*\u001b[01;31m\u001b[K>\u001b[m\u001b[K data);\n",
            "                                      \u001b[01;31m\u001b[K^\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:38:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 2 is invalid\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:21:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[Kstd::vector\u001b[m\u001b[K’ is not a type\n",
            "     void train(std::\u001b[01;31m\u001b[Kvector\u001b[m\u001b[K<RowVector*> data);\n",
            "                     \u001b[01;31m\u001b[K^~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:30:27:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Kexpected ‘\u001b[01m\u001b[K,\u001b[m\u001b[K’ or ‘\u001b[01m\u001b[K...\u001b[m\u001b[K’ before ‘\u001b[01m\u001b[K<\u001b[m\u001b[K’ token\n",
            "     void train(std::vector\u001b[01;31m\u001b[K<\u001b[m\u001b[KRowVector*> data);\n",
            "                           \u001b[01;31m\u001b[K^\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:39:17:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KRowVector\u001b[m\u001b[K’ was not declared in this scope\n",
            "     std::vector<\u001b[01;31m\u001b[KRowVector\u001b[m\u001b[K*> neuronLayers; // stores the different layers of out network\n",
            "                 \u001b[01;31m\u001b[K^~~~~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:39:27:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 1 is invalid\n",
            "     std::vector<RowVector*\u001b[01;31m\u001b[K>\u001b[m\u001b[K neuronLayers; // stores the different layers of out network\n",
            "                           \u001b[01;31m\u001b[K^\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:39:27:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 2 is invalid\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:40:17:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KRowVector\u001b[m\u001b[K’ was not declared in this scope\n",
            "     std::vector<\u001b[01;31m\u001b[KRowVector\u001b[m\u001b[K*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers\n",
            "                 \u001b[01;31m\u001b[K^~~~~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:40:27:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 1 is invalid\n",
            "     std::vector<RowVector*\u001b[01;31m\u001b[K>\u001b[m\u001b[K cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers\n",
            "                           \u001b[01;31m\u001b[K^\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:40:27:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 2 is invalid\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:41:17:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KRowVector\u001b[m\u001b[K’ was not declared in this scope\n",
            "     std::vector<\u001b[01;31m\u001b[KRowVector\u001b[m\u001b[K*> deltas; // stores the error contribution of each neurons\n",
            "                 \u001b[01;31m\u001b[K^~~~~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:41:27:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 1 is invalid\n",
            "     std::vector<RowVector*\u001b[01;31m\u001b[K>\u001b[m\u001b[K deltas; // stores the error contribution of each neurons\n",
            "                           \u001b[01;31m\u001b[K^\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:41:27:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 2 is invalid\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:42:17:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[K‘\u001b[01m\u001b[KMatrix\u001b[m\u001b[K’ was not declared in this scope\n",
            "     std::vector<\u001b[01;31m\u001b[KMatrix\u001b[m\u001b[K*> weights; // the connection weights itself\n",
            "                 \u001b[01;31m\u001b[K^~~~~~\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:42:17:\u001b[m\u001b[K \u001b[01;36m\u001b[Knote: \u001b[m\u001b[Ksuggested alternative: ‘\u001b[01m\u001b[Katoi\u001b[m\u001b[K’\n",
            "     std::vector<\u001b[01;36m\u001b[KMatrix\u001b[m\u001b[K*> weights; // the connection weights itself\n",
            "                 \u001b[01;36m\u001b[K^~~~~~\u001b[m\u001b[K\n",
            "                 \u001b[32m\u001b[Katoi\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:42:24:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 1 is invalid\n",
            "     std::vector<Matrix*\u001b[01;31m\u001b[K>\u001b[m\u001b[K weights; // the connection weights itself\n",
            "                        \u001b[01;31m\u001b[K^\u001b[m\u001b[K\n",
            "\u001b[01m\u001b[KNeuralNetwork.cpp:42:24:\u001b[m\u001b[K \u001b[01;31m\u001b[Kerror: \u001b[m\u001b[Ktemplate argument 2 is invalid\n"
          ]
        }
      ]
    }
  ]
}