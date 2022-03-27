# Label conditioned segmentation (LCS)

Code for our MIDL 2022 paper: *Label conditioned segmentation* [https://arxiv.org/abs/2203.10091]  

An LCS model outputs a single-channel segmentation map regardless of how many classes are used for training. The output class is conditioned on an additional input provided to the model.

Because the size of the model output is independent of the number of target classes, our method can handle segmentation tasks with very large image size and a large number of classes in a single model.

Similar to many one-shot learning methods, LCS can produce previously unseen labels during inference time without further training.

<img src="https://github.com/tym002/Label-conditioned-segmentation/blob/main/architecture_final.png" width="600">

## requirements: 

`tensorflow-gpu 1.15.0`

`python 3.6.13`

## Code:

`python main.py` to train or test model 

In line 174, `mode = train` or `mode = predict`

Input shape: `(Batch, x, y, z, channel)` for 3D image

`generator.py` contains the generator that samples random atlas label classes

## Citation:
If you find our code useful, please cite our work, thank you!

`@article{ma2022label,
  title={Label conditioned segmentation},
  author={Ma, Tianyu and Lee, Benjamin C and Sabuncu, Mert R},
  journal={arXiv preprint arXiv:2203.10091},
  year={2022}
}`


