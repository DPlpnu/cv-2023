from PIL import Image
import numpy as np

im = Image.open("monkey4.jpg")
im.show()

im_gray = im.convert('L')

im_arr = np.array(im_gray)

kernels_kirsch = [
    np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
    np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
    np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
    np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
    np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
    np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
    np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
    np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
]

kernel_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

result_kirsch = np.zeros_like(im_arr)
for i in range(1, im_arr.shape[0]-1):
    for j in range(1, im_arr.shape[1]-1):
        max_val = 0
        for kernel in kernels_kirsch:
            val = np.sum(np.abs(kernel * im_arr[i-1:i+2, j-1:j+2]))
            if val > max_val:
                max_val = val
        result_kirsch[i, j] = max_val

im_kirsch = Image.fromarray(result_kirsch.astype(np.uint8))

result_laplacian = np.zeros_like(im_arr)
for i in range(1, im_arr.shape[0]-1):
    for j in range(1, im_arr.shape[1]-1):
        val = np.sum(kernel_laplacian * im_arr[i-1:i+2, j-1:j+2])
        result_laplacian[i, j] = np.abs(val)

im_laplacian = Image.fromarray(result_laplacian.astype(np.uint8))

Image.fromarray(np.hstack((im_arr, result_kirsch, result_laplacian)).astype(np.uint8)).show()
