这个工具和文章《Qualitative and Quantitative Model Checking against Recurrent Neural Networks》所对应。

以poly命名的相关文件实现了polyhedra的H-presentation和V-presentation转换，体积计算，包含关系，线性变换，relu函数等功能；

network_computation.py 构造不同时刻的权重矩阵和偏置向量；

forward（backward）propagation则是完整的前向传播和后向传播的完整流程实现；

qq_verify.py给出了定型验证和定量验证的验证方法；

以EXM命名的文件实现了文章中实验部分的验证功能。

此外，vertex_reduce.py 则是文中提到的顶点压缩算法的具体实现。test.py给出了一个简单可验证的前向传播的例子。