# TransH
Reimplement TransH mode

# file 
TransH.py _The initial version which use my own code to implement.  
TransH_pytorch.py _Try to use the autograd function of pytorch to calculate the gradient of each embedding.    
TransH_torch.py _Use the torch.nn.Embedding and marginRankingLoss function to reimplement the TransH model.  

TransH_torch.py 中完全使用pytorch实现transH模型，以及测试功能，在FB15K 以及 WN18两个数据集上的Link prediction测试结果效果基本达到论文原文中的测试结果

# TransH_torch.py Test_result

WN18:
|    |raw|fil|          
|----|----|----|
|	   | MR    hit10  | MR    hit10 | 
|TransH| 253   0.6719 | 242    0.768 |  

FB15K:
|    |raw |fil|
|----|:----:|:----:|
|	    | MR      hit10 | MR     hit10 |
|TransH |327      0.457 | 188    0.663 |
