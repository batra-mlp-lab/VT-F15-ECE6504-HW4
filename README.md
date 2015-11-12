# [ECE 6504 Deep Learning for Perception][1]

## Homework 4

In this homework, we will implement an LSTM model for Visual Question Answering. This homework is based on the papers, [Exploring Models and Data for Image Question Answering][2] and [VQA: Visual Question Answering][4]. You are free to use any Torch library. We recommend using [Element-Research/rnn][3].

The dataset provided consists of three types of questions: 'what color','what is on the' and 'what sport is'. The data is processed and provided in a torch readable format in `data_HW4.t7`.

Download the starter code [here](https://github.com/batra-mlp-lab/VT-F15-ECE6504-HW4/archive/1.0.zip).

### Q1: Blind QA model (20 points)

In this part you need to implement a blind model i.e. the model does not have inputs from the image. Fill in the model architecture details in `train_qa.lua`.

### Q2: VQA model (15 points)

In this part, you need to augment the blind model to take inputs from the image. The `fc7` features of the images can be downloaded [here](https://filebox.ece.vt.edu/~f15ece6504/fc7_hw4.t7). The fc7 features are in lua table format. Each feature vector is referenced against it's image id. Fill in the model architecture details in `train_vqa.lua`.

### Q3: Try something extra (Up to 15 points)

This part is for you to implement something extra. Some pointers:

- VQA for other question types. Code to prepare the data is in `fetchQA.py` and `fetchData.lua`
- different architectures for VQA

**Deliverables**

- Zip containing the following:

  - Completed files: `train_qa.lua` and `train_vqa.lua`
  - Results for Q1 and Q2 on the test set - `data_HW4_test`
  - Code for Q3
  - README with results of all the parts and a brief explanation of Q3

References:

1. [Exploring Models and Data for Image Question Answering][2], Ren et al., NIPS15
2. [VQA: Visual Question Answering][4], Antol et al., ICCV15

[1]: https://computing.ece.vt.edu/~f15ece6504/
[2]: http://arxiv.org/abs/1505.02074
[3]: https://github.com/Element-Research/rnn
[4]: http://visualqa.org/VQA_ICCV2015.pdf

---

&#169; 2015 Virginia Tech
