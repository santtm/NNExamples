[Neural Networks Part 8](https://www.youtube.com/watch?v=HGwBXDKFk9I&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=14)

<pre>
Input: X
 _____________ 
| ■         ■ |
|   ■     ■   |
|     ■ ■     |
|     ■ ■     |
|   ■     ■   |
| ■         ■ |
 ‾‾‾‾‾‾‾‾‾‾‾‾‾ 

Prediction: X
Max, Argmax:
X: 1.000, O: 0.000

Prediction: X
Max, Softmax:
X: 0.731, O: 0.269

Prediction: X
Mean, Argmax:
X: 1.000, O: 0.000

Prediction: X
Mean, Softmax:
X: 0.731, O: 0.269


Input: X_1
 _____________ 
| ■         ■ |
|   ■     ■   |
|     ■       |
|       ■     |
|     ■   ■   |
| ■         ■ |
 ‾‾‾‾‾‾‾‾‾‾‾‾‾ 

Prediction: X
Max, Argmax:
X: 1.000, O: 0.000

Prediction: X
Max, Softmax:
X: 0.731, O: 0.269

Prediction: X
Mean, Argmax:
X: 1.000, O: 0.000

Prediction: X
Mean, Softmax:
X: 0.731, O: 0.269


Input: X_2
 _____________
| ■       ■   |
|   ■   ■     |
|         ■   |
|         ■   |
|   ■   ■     |
| ■       ■   |
 ‾‾‾‾‾‾‾‾‾‾‾‾‾

Prediction: X
Max, Argmax:
X: 1.000, O: 0.000

Prediction: X
Max, Softmax:
X: 0.731, O: 0.269

Prediction: X
Mean, Argmax:
X: 1.000, O: 0.000

Prediction: X
Mean, Softmax:
X: 0.731, O: 0.269


Input: X_3
 _____________
| ■       ■   |
|   ■   ■     |
|     ■       |
|   ■   ■     |
| ■       ■   |
|             |
 ‾‾‾‾‾‾‾‾‾‾‾‾‾

Prediction: O
Max, Argmax:
X: 0.000, O: 1.000

Prediction: O
Max, Softmax:
X: 0.269, O: 0.731

Prediction: X
Mean, Argmax:
X: 1.000, O: 0.000

Prediction: X
Mean, Softmax:
X: 0.731, O: 0.269


Input: X_4
 _____________
| ■       ■   |
|   ■   ■     |
|     ■       |
|   ■   ■     |
| ■       ■   |
| ■         ■ |
 ‾‾‾‾‾‾‾‾‾‾‾‾‾

Prediction: O
Max, Argmax:
X: 0.000, O: 1.000

Prediction: O
Max, Softmax:
X: 0.269, O: 0.731

Prediction: X
Mean, Argmax:
X: 1.000, O: 0.000

Prediction: X
Mean, Softmax:
X: 0.731, O: 0.269


Input: O
 _____________
|     ■ ■     |
|   ■     ■   |
| ■         ■ |
| ■         ■ |
|   ■     ■   |
|     ■ ■     |
 ‾‾‾‾‾‾‾‾‾‾‾‾‾

Prediction: O
Max, Argmax:
X: 0.000, O: 1.000

Prediction: O
Max, Softmax:
X: 0.269, O: 0.731

Prediction: X
Mean, Argmax:
X: 1.000, O: 0.000

Prediction: X
Mean, Softmax:
X: 0.731, O: 0.269


Input: O_1
 _____________
|   ■ ■       |
| ■     ■ ■   |
|           ■ |
|           ■ |
| ■     ■ ■   |
|   ■ ■       |
 ‾‾‾‾‾‾‾‾‾‾‾‾‾

Prediction: X
Max, Argmax:
X: 1.000, O: 0.000

Prediction: X
Max, Softmax:
X: 0.731, O: 0.269

Prediction: X
Mean, Argmax:
X: 1.000, O: 0.000

Prediction: X
Mean, Softmax:
X: 0.731, O: 0.269


Input: O_2
 _____________
|       ■ ■   |
|     ■     ■ |
|   ■         |
|   ■         |
|     ■     ■ |
|       ■ ■   |
 ‾‾‾‾‾‾‾‾‾‾‾‾‾

Prediction: O
Max, Argmax:
X: 0.000, O: 1.000

Prediction: O
Max, Softmax:
X: 0.269, O: 0.731

Prediction: X
Mean, Argmax:
X: 1.000, O: 0.000

Prediction: X
Mean, Softmax:
X: 0.731, O: 0.269


Input: O_3
 _____________
|   ■ ■       |
| ■     ■ ■   |
|         ■   |
|         ■   |
| ■     ■ ■   |
|   ■ ■       |
 ‾‾‾‾‾‾‾‾‾‾‾‾‾

Prediction: X
Max, Argmax:
X: 1.000, O: 0.000

Prediction: X
Max, Softmax:
X: 0.731, O: 0.269

Prediction: X
Mean, Argmax:
X: 1.000, O: 0.000

Prediction: X
Mean, Softmax:
X: 0.731, O: 0.269


Input: O_4
 _____________
|   ■ ■       |
| ■     ■ ■   |
|           ■ |
|         ■   |
| ■     ■     |
|   ■ ■       |
 ‾‾‾‾‾‾‾‾‾‾‾‾‾

Prediction: X
Max, Argmax:
X: 1.000, O: 0.000

Prediction: X
Max, Softmax:
X: 0.731, O: 0.269

Prediction: X
Mean, Argmax:
X: 1.000, O: 0.000

Prediction: X
Mean, Softmax:
X: 0.731, O: 0.269
</pre>
