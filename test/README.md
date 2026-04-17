# Supervised-manifold learning for Conformer Generation


* 基于分子表示得到的Q图（概率）和基于Y的Q图（概率），让二者CE最小化,然后更新Y，如果Q==Q(Y),那么，得到的Y应该构象应该一致。
* 添加类Parametric UMAP的GNN或者MLP学习层
* 如何做到摊销推理？当前是一个分子的推理，是否可以在合适的地方加入共享的可学习的一组参数？
