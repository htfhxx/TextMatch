# 基于Pytorch的中文语义相似度匹配模型ABCNN


运行环境：
```angular2html
python3.7
pytorch1.2
transformers2.5.1
```

训练 & 测试
```angular2html
python3 train.py --train_dir --max_len 50 --epochs 100 --batch_size 256 --patience 10 --gpu_index 0 

python3 test.py --max_len 128 --batch_size 256 --gpu_index 0 
```


测试集结果对比：  

| model | LCQMC | BQ |
| :-----| ----: | :----: |
| BERT-base-Chinese |  |  |
| ABCNN - conv_num-1 - max_len-50 |  |  |


参考代码：
```angular2html
https://github.com/htfhxx/TextMatch/ (主要参考)
https://github.com/htfhxx/abcnn_pytorch
https://github.com/htfhxx/TextMatching-Chinese
https://github.com/htfhxx/FinetuneZoo
```