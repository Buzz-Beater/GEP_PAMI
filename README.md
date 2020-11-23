# GEP

This repo is adapted from the [original GEP repo](https://github.com/SiyuanQi/generalized-earley-parser) and contains code and adjustments for our TPAMI 2020 paper.

[A Generalized Earley Parser for Human Activity Parsing and Prediction](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9018126)

Siyuan Qi, Baoxiong Jia, Siyuan Huang, Ping Wei, and Song-Chun Zhu

*IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2020


# Dependencies

Please check that all required packages from ```requirements.txt``` are properly installed.


# Experiments

This repo contains code for reproducing the results reported in our TPAMI paper.

To run your experments properly, please download the datasets and adjust the paths information properly in ```config.py```.

We provide three example scripts for showing how to use this code for the purpose of activity parsing and also future prediction.

First, we show how to run experiments for activity parsing in ```breakfast_det.sh``` and ```gep_breakfast_det.sh```. These two shell scripts run ```baseline``` and ```gep``` for recognizing human actions respectively. As the breakfast dataset is big in frame number, we tried subsampling frames as one hyper-parameter which could be tuned during experiment. Please change the ```LOG_PATH``` to your correct logging path for storing the results before running the scripts.

Next, for activity prediction, we use prediction on CAD dataset as an example. As shown in ```cad_pred.sh```, we run baseline training/eval and also gep prediction. We report and store models' performance under different prediction duration, which could be set in the shell script. Please also change the ```LOG_PATH``` to your correct loggging path for storing the results.

# Data
For features and grammar files used for reproducing experimental results, please find at [here](https://drive.google.com/drive/folders/1_3rr3O1AtbZsGHwy33JPkSSQAOzq5Z8j?usp=sharing). Please put the unzipped directory at a valid location and fix path configurations inside ```config.py``` to match the usage of features path used in ```datasets/{dataset}.py```.



# Citation

If you find the paper and/or the code helpful, please cite
```
@inproceedings{qi2018future,
    title={Generalized Earley Parser: Bridging Symbolic Grammars and Sequence Data for Future Prediction},
    author={Qi, Siyuan and Jia, Baoxiong and Zhu, Song-Chun},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2018}
}
@article{qi2020generalized,
  title={A Generalized Earley Parser for Human Activity Parsing and Prediction},
  author={Qi, Siyuan and Jia, Baoxiong and Huang, Siyuan and Wei, Ping and Zhu, Song-Chun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```

