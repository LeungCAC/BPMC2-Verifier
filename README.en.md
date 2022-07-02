This toolkit corresponds to the paper 《Qualitative and Quantitative Model Checking against Recurrent Neural Networks》.

The related files named with "poly_****" realize the functions of polyhedra, such as the conversion between h-presentation and v-presentation, volume computation, inclusion relationship judgement, linear transformation, relu function and so on;

"network_ computation.py" constructs the weight matrix and bias vector at different times;

"forward (resp. backward)_propagation.py" is a complete process realization of forward (resp. backward) propagation;

"qq_ verify.py" gives the verification methods of qualitative verification and quantitative verification;

The file named like "EXM***.py" realizes the verification function of the experimental part in the paper;

In addition, "vertex_ reduce.py" is the concrete implementation of the vertex compression algorithm mentioned in this paper and "test.py" gives a simple and verifiable example of forward propagation.