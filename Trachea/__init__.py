# listen listen

## Transformers 
"""
consists of an encoder and a decoder 
(x1,x2,x3,x4,....,xn) --> encoder --> (z1,z2,z3,z4,.....zn)
(z1,z2,z3,z4,.....,zn) --> decoder --> (y1,y2,y3,y4,.....,ym)

encoder -->
    composed of a stack of N = 6 identical layers   
    each layer has 2 sub layers 
        1. multi headed self attention mech
        2. simple fully connected
    uses a residual connnection around every 2 sublayers
    followed by layer norm
    output of every sublayer is LayerNorm(x+Sublayer(x)) --> d(model) = 512

decoder --> 
    also composed of N identical layers 
    with 3 sub layers additional sub layer performs multi head attention over the output of the sub layers
    and Layer Norm

Attention --> 
    mapping a query and a set of key-value pairs to an output (every thing is a ventor)

    takes in query,key,values

Dot product attention -->
    compute dot product of the query with all the keys and divide each by the root of |dimention of keys|
    every thing is packed into a matrix Q

    Attention(Q,K,V) = softmax(Q*K^T / root|dim k|)*V


"""
