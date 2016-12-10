/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fasttext.h"

#include <fenv.h>

#include <iostream>
#include <iomanip>
#include <thread>
#include <algorithm>

namespace fasttext {

    void FastText::getVector(Vector &vec, const std::string &word) {
        const std::vector<int32_t> &ngrams = dict_->getNgrams(word);
        vec.zero();
        for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
            vec.addRow(*input_, *it);
        }
        if (ngrams.size() > 0) {
            vec.mul(1.0 / ngrams.size());
        }
    }

    void FastText::saveVectors() {
        std::ofstream ofs(args_->output + ".vec");
        if (!ofs.is_open()) {
            std::cout << "Error opening file for saving vectors." << std::endl;
            exit(EXIT_FAILURE);
        }
        ofs << dict_->nwords() << " " << args_->dim << std::endl;
        Vector vec(args_->dim);
        for (int32_t i = 0; i < dict_->nwords(); i++) {
            std::string word = dict_->getWord(i);
            getVector(vec, word);
            ofs << word << " " << vec << std::endl;
        }
        ofs.close();
    }

    void FastText::saveModel() {
        std::ofstream ofs(args_->output + ".bin", std::ofstream::binary);
        if (!ofs.is_open()) {
            std::cerr << "Model file cannot be opened for saving!" << std::endl;
            exit(EXIT_FAILURE);
        }
        args_->save(ofs);
        dict_->save(ofs);
        input_->save(ofs);
        output_->save(ofs);
        ofs.close();
    }

    void FastText::loadModel(const std::string &filename) {
        std::ifstream ifs(filename, std::ifstream::binary);
        if (!ifs.is_open()) {
            std::cerr << "Model file cannot be opened for loading!" << std::endl;
            exit(EXIT_FAILURE);
        }
        loadModel(ifs);
        ifs.close();
    }

    void FastText::loadModel(std::istream &in) {
        args_ = std::make_shared<Args>();
        dict_ = std::make_shared<Dictionary>(args_);
        input_ = std::make_shared<Matrix>();
        output_ = std::make_shared<Matrix>();
        args_->load(in);
        dict_->load(in);
        input_->load(in);
        output_->load(in);
        model_ = std::make_shared<Model>(input_, output_, args_, 0);
        if (args_->model == model_name::sup) {
            model_->setTargetCounts(dict_->getCounts(entry_type::label));
        } else {
            model_->setTargetCounts(dict_->getCounts(entry_type::word));
        }
    }

    void FastText::printInfo(real progress, real loss) {
        real t = real(clock() - start) / CLOCKS_PER_SEC;
        real wst = real(tokenCount) / t;
        real lr = args_->lr * (1.0 - progress);
        int eta = int(t / progress * (1 - progress) / args_->thread);
        int etah = eta / 3600;
        int etam = (eta - etah * 3600) / 60;
        std::cout << std::fixed;
        std::cout << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
        std::cout << "  words/sec/thread: " << std::setprecision(0) << wst;
        std::cout << "  lr: " << std::setprecision(6) << lr;
        std::cout << "  loss: " << std::setprecision(6) << loss;
        std::cout << "  eta: " << etah << "h" << etam << "m ";
        std::cout << std::flush;
    }

    void FastText::supervised(Model &model, real lr,
                              const std::vector<int32_t> &line,
                              const std::vector<int32_t> &labels) {
        if (labels.size() == 0 || line.size() == 0) return;
        /*因为一个句子可以打上多个 label，但是 fastText 的架构实际上只有支持一个 label,
         * 所以这里随机选择一个 label 来更新模型，这样做会让其它 label 被忽略*/
        std::uniform_int_distribution<> uniform(0, labels.size() - 1);
        int32_t i = uniform(model.rng);
        model.update(line, labels[i], lr);
    }

    void FastText::cbow(Model &model, real lr,
                        const std::vector<int32_t> &line) {
        std::vector<int32_t> bow;
        std::uniform_int_distribution<> uniform(1, args_->ws);
        /*在一个句子中，每个词可以进行一次update*/
        for (int32_t w = 0; w < line.size(); w++) {
            int32_t boundary = uniform(model.rng);  /*一个词语的上下文长度是随机产生的*/
            bow.clear();
            /*将窗口内的词语加入input中*/
            for (int32_t c = -boundary; c <= boundary; c++) {
                /*防止数组越界*/
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    /*加入input的词语除了文本中的词语还有n-gram*/
                    const std::vector<int32_t> &ngrams = dict_->getNgrams(line[w + c]);
                    bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
                }
            }
            /*完成一次CBOW更新*/
            model.update(bow, line[w], lr);
        }
    }

    void FastText::skipgram(Model &model, real lr,
                            const std::vector<int32_t> &line) {
        std::uniform_int_distribution<> uniform(1, args_->ws);
        for (int32_t w = 0; w < line.size(); w++) {
            int32_t boundary = uniform(model.rng);
            const std::vector<int32_t> &ngrams = dict_->getNgrams(line[w]);
            for (int32_t c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    model.update(ngrams, line[w + c], lr);
                }
            }
        }
    }

    void FastText::test(std::istream &in, int32_t k) {
        int32_t nexamples = 0, nlabels = 0;
        double precision = 0.0;
        std::vector<int32_t> line, labels;

        while (in.peek() != EOF) {
            dict_->getLine(in, line, labels, model_->rng);
            dict_->addNgrams(line, args_->wordNgrams);
            if (labels.size() > 0 && line.size() > 0) {
                std::vector<std::pair<real, int32_t>> modelPredictions;
                model_->predict(line, k, modelPredictions);
                for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
                    if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
                        precision += 1.0;
                    }
                }
                nexamples++;
                nlabels += labels.size();
            }
        }
        std::cout << std::setprecision(3);
        std::cout << "P@" << k << ": " << precision / (k * nexamples) << std::endl;
        std::cout << "R@" << k << ": " << precision / nlabels << std::endl;
        std::cout << "Number of examples: " << nexamples << std::endl;
    }

    void FastText::predict(std::istream &in, int32_t k,
                           std::vector<std::pair<real, std::string>> &predictions) const {
        std::vector<int32_t> words, labels;
        dict_->getLine(in, words, labels, model_->rng);
        dict_->addNgrams(words, args_->wordNgrams);   /*将一个词的 n-gram 加入词表，用于处理未登录词*/
        if (words.empty()) return;
        Vector hidden(args_->dim);
        Vector output(dict_->nlabels());
        std::vector<std::pair<real, int32_t>> modelPredictions;
        model_->predict(words, k, modelPredictions, hidden, output);
        predictions.clear();
        for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
            predictions.push_back(std::make_pair(it->first, dict_->getLabel(it->second)));
        }
    }

    void FastText::predict(std::istream &in, int32_t k, bool print_prob) {
        std::vector<std::pair<real, std::string>> predictions;
        while (in.peek() != EOF) {
            predict(in, k, predictions);
            if (predictions.empty()) {
                std::cout << "n/a" << std::endl;
                continue;
            }
            for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
                if (it != predictions.cbegin()) {
                    std::cout << ' ';
                }
                std::cout << it->second;
                if (print_prob) {
                    std::cout << ' ' << exp(it->first);
                }
            }
            std::cout << std::endl;
        }
    }

    void FastText::wordVectors() {
        std::string word;
        Vector vec(args_->dim);
        while (std::cin >> word) {
            getVector(vec, word);
            std::cout << word << " " << vec << std::endl;
        }
    }

    void FastText::textVectors() {
        std::vector<int32_t> line, labels;
        Vector vec(args_->dim);
        while (std::cin.peek() != EOF) {
            dict_->getLine(std::cin, line, labels, model_->rng);
            dict_->addNgrams(line, args_->wordNgrams);
            vec.zero();
            for (auto it = line.cbegin(); it != line.cend(); ++it) {
                vec.addRow(*input_, *it);
            }
            if (!line.empty()) {
                vec.mul(1.0 / line.size());
            }
            std::cout << vec << std::endl;
        }
    }

    void FastText::printVectors() {
        if (args_->model == model_name::sup) {
            textVectors();
        } else {
            wordVectors();
        }
    }

    void FastText::trainThread(int32_t threadId) {
        std::ifstream ifs(args_->input);
        utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);    /*根据线程数将文件分块读入*/

        Model model(input_, output_, args_, threadId);
        if (args_->model == model_name::sup) {
            model.setTargetCounts(dict_->getCounts(entry_type::label));
        } else {
            model.setTargetCounts(dict_->getCounts(entry_type::word));
        }

        const int64_t ntokens = dict_->ntokens();     /*训练文件中的token数*/
        int64_t localTokenCount = 0;  /*当前线程处理完毕后的token数*/
        std::vector<int32_t> line, labels;
        while (tokenCount < args_->epoch * ntokens) { /*tokenCount是所有线程处理完毕的token总数*/
            real progress = real(tokenCount) / (args_->epoch * ntokens);    /*训练完成进度,取值在0~1*/
            real lr = args_->lr * (1.0 - progress);     /*学习率随着完成进度增加而下降*/
            localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
            /*根据训练需求的不同，更新策略发生变化*/
            if (args_->model == model_name::sup) {  /*有监督学习（分类）*/
                dict_->addNgrams(line, args_->wordNgrams);
                supervised(model, lr, line, labels);
            } else if (args_->model == model_name::cbow) {  /*word2vec(CBOW)*/
                cbow(model, lr, line);
            } else if (args_->model == model_name::sg) {  /*word2vec(skipgram)*/
                skipgram(model, lr, line);
            }
            if (localTokenCount > args_->lrUpdateRate) {
                tokenCount += localTokenCount;
                localTokenCount = 0;
                if (threadId == 0 && args_->verbose > 1) {
                    printInfo(progress, model.getLoss());
                }
            }
        }
        if (threadId == 0 && args_->verbose > 0) {
            printInfo(1.0, model.getLoss());
            std::cout << std::endl;
        }
        ifs.close();
    }

    void FastText::loadVectors(std::string filename) {
        std::ifstream in(filename);
        std::vector<std::string> words;
        std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
        int64_t n, dim;
        if (!in.is_open()) {
            std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }
        in >> n >> dim;
        if (dim != args_->dim) {
            std::cerr << "Dimension of pretrained vectors does not match -dim option"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        mat = std::make_shared<Matrix>(n, dim);
        for (size_t i = 0; i < n; i++) {
            std::string word;
            in >> word;
            words.push_back(word);
            dict_->add(word);
            for (size_t j = 0; j < dim; j++) {
                in >> mat->data_[i * dim + j];
            }
        }
        in.close();

        dict_->threshold(1, 0);
        input_ = std::make_shared<Matrix>(dict_->nwords() + args_->bucket, args_->dim);
        input_->uniform(1.0 / args_->dim);

        for (size_t i = 0; i < n; i++) {
            int32_t idx = dict_->getId(words[i]);
            if (idx < 0 || idx >= dict_->nwords()) continue;
            for (size_t j = 0; j < dim; j++) {
                input_->data_[idx * dim + j] = mat->data_[i * dim + j];
            }
        }
    }

    void FastText::train(std::shared_ptr<Args> args) {
        args_ = args;
        dict_ = std::make_shared<Dictionary>(args_);
        if (args_->input == "-") {
            // manage expectations
            std::cerr << "Cannot use stdin for training!" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::ifstream ifs(args_->input);
        if (!ifs.is_open()) {
            std::cerr << "Input file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }
        /*根据输入文件初始化词典*/
        dict_->readFromFile(ifs);
        ifs.close();

        /*初始化输入层，对于普通word2vec，输入层是一个词向量的查找表,
         * fastText 用了word n-gram 作为输入，所以输入矩阵的大小为 (nwords + ngram 种类) * dim
         * 代码中，所有 word n-gram 都被 hash 到固定数目的 bucket 中，所以输入矩阵的大小为(nwords + bucket 个数) * dim
         * */
        if (args_->pretrainedVectors.size() != 0) {
            loadVectors(args_->pretrainedVectors);
        } else {
            input_ = std::make_shared<Matrix>(dict_->nwords() + args_->bucket, args_->dim);
            input_->uniform(1.0 / args_->dim);
        }

        if (args_->model == model_name::sup) {
            output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim);   /*训练分类器*/
        } else {
            output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);    /*训练词向量*/
        }
        output_->zero();

        start = clock();
        tokenCount = 0;
        std::vector<std::thread> threads;
        for (int32_t i = 0; i < args_->thread; i++) {
            threads.push_back(std::thread([=]() { trainThread(i); }));
        }
        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
        model_ = std::make_shared<Model>(input_, output_, args_, 0);

        saveModel();
        if (args_->model != model_name::sup) {
            saveVectors();
        }
    }

}
