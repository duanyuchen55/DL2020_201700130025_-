import numpy as np
a = np.array([
              [
                  [1, 5, 5, 2],
                  [9, -6, 2, 8],
                  [-3, 7, -9, 1]
              ],

              [
                  [-1, 5, -5, 2],
                  [9, 6, 2, 8],
                  [3, 7, 9, 1]
              ]
            ])
# print(np.sum(a, axis=0))

b = np.array([
              [
                  [1, 5, 5, 2],
                  [9, -6, 2, 8],
                  [-3, 7, -9, 1]
              ],

              [
                  [-1, 5, -5, 2],
                  [9, 6, 2, 8],
                  [3, 7, 9, 1]
              ]
            ])
# print(np.sum(b, axis=0, keepdims=True))


c = b[...]

print(c)