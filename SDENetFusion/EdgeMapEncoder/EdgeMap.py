import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_structure_tensor_edges(img):
    # Ensure image is float32 for precision
    img = img.astype(np.float32)
    
    # 1. Compute spatial derivatives for all channels simultaneously
    # If img shape is (H, W, 3), Sobel processes each channel independently
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    # 2. Compute the components of the Color Structure Tensor (summing over the color axis)
    Jxx = np.sum(Ix ** 2, axis=2)
    Jyy = np.sum(Iy ** 2, axis=2)
    Jxy = np.sum(Ix * Iy, axis=2)
    # print(f"Jxx shape: {Jxx.shape}, Jyy shape: {Jyy.shape}, Jxy shape: {Jxy.shape}")
    # Optional: Apply a Gaussian blur to Jxx, Jyy, Jxy if you want a local window integration
    Jxx = cv2.GaussianBlur(Jxx, (3,3), 0)
    Jyy = cv2.GaussianBlur(Jyy, (3,3), 0)
    Jxy = cv2.GaussianBlur(Jxy, (3,3), 0)

    # 3. Calculate eigenvalues of the 2x2 matrix at each pixel
    # The larger eigenvalue corresponds to the edge strength
    # Formula for eigenvalues of a 2x2 symmetric matrix: 
    # λ = 0.5 * ((Jxx + Jyy) ± sqrt((Jxx - Jyy)^2 + 4*Jxy^2))
    trace = Jxx + Jyy
    
    # lambda1 = 0.5 * (trace + np.sqrt(np.maximum(0, (Jxx - Jyy)**2 + 4 * Jxy**2)))
    # E = np.sqrt(lambda1)  # Edge magnitude is often taken as the square root of the maximum eigenvalue
    # det = Jxx * Jyy - Jxy ** 2
    # discriminant = np.sqrt(np.maximum(0, (Jxx - Jyy)**2 + 4 * Jxy**2))
    
    # lambda_max = 0.5 * (trace + discriminant)
    
    # # Edge magnitude is often taken as the square root of the maximum eigenvalue
    # edge_magnitude = np.sqrt(lambda_max)
    
    # # Normalize for visualization
    # edge_magnitude = cv2.normalize(edge_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return trace, Jxx, Jyy, Jxy
    # return E

def edges_from_grads(img):
    # Ensure image is float32 for precision
    img = img.astype(np.float32)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Optional: smooth the image to reduce noise
    # Compute spatial derivatives for all channels simultaneously
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute the gradient magnitude (summing over the color axis)
    grad_magnitude = np.sqrt(np.sum(Ix ** 2 + Iy ** 2, axis=2))
    return grad_magnitude

def remap_edges(img):
    image = img.astype(np.float32).copy()

    min_value = image.min()
    max_value = image.max()

    return (image - min_value) / (
        max_value - min_value + 1e-8
    )

if __name__ == "__main__":
    # Usage
    image = cv2.imread('C:\\Users\\gadda\\Documents\\FeatureFusionMoNuSAC\\SDENet-Fusion\\GeometricEncoder\\slicSamples\\image.png')
    trace, Jxx, Jyy, Jxy = color_structure_tensor_edges(image)
    sobel_edges = edges_from_grads(image)
    plt.subplot(1, 3, 1)
    plt.imshow(image[:,:,::-1], cmap='gray')
    plt.title('Original Image')
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(remap_edges(trace), cmap='gray')
    plt.title('Color Structure Tensor Edges (trace)')
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(remap_edges(sobel_edges), cmap='gray')
    plt.title('Gradient Magnitude Edges')
    plt.axis("off")
    plt.show()

    plt.subplot(1,3,1)
    plt.imshow(remap_edges(Jxx), cmap='gray')
    plt.title('Jxx')
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(remap_edges(Jyy), cmap='gray')
    plt.title('Jyy')
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(remap_edges(Jxy), cmap='gray')
    plt.title('Jxy')
    plt.axis("off")
    plt.show()