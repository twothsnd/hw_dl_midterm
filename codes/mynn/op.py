from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim)) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        scale = np.sqrt(2.0 / (in_channels * self.kernel_size * self.kernel_size))
        self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size, self.kernel_size)) * scale
        self.b = np.zeros((1, out_channels, 1, 1))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None
        self.input_padded_shape = None
        self.cols = None
        self.out_hw = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        self.input = X
        n, c, h, w = X.shape
        k = self.kernel_size
        out_h = (h + 2 * self.padding - k) // self.stride + 1
        out_w = (w + 2 * self.padding - k) // self.stride + 1
        self.out_hw = (out_h, out_w)

        if self.padding > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            X_pad = X
        self.input_padded_shape = X_pad.shape
        cols = self._im2col(X_pad, k, self.stride, out_h, out_w)
        self.cols = cols
        W_col = self.W.reshape(self.out_channels, -1)
        out = cols @ W_col.T + self.b.reshape(1, self.out_channels)
        out = out.reshape(n, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.input is not None and self.cols is not None and self.out_hw is not None
        n = grads.shape[0]
        k = self.kernel_size
        out_h, out_w = self.out_hw
        grad_out = grads.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.grads['W'] = (grad_out.T @ self.cols).reshape(self.W.shape)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        W_col = self.W.reshape(self.out_channels, -1)
        dcols = grad_out @ W_col
        dX_pad = self._col2im(dcols, self.input_padded_shape, k, self.stride, out_h, out_w)
        if self.padding > 0:
            return dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dX_pad

    @staticmethod
    def _im2col(X, kernel_size, stride, out_h, out_w):
        n, c, _, _ = X.shape
        cols = np.empty((n, out_h, out_w, c, kernel_size, kernel_size), dtype=X.dtype)
        for i in range(kernel_size):
            i_end = i + stride * out_h
            for j in range(kernel_size):
                j_end = j + stride * out_w
                cols[:, :, :, :, i, j] = X[:, :, i:i_end:stride, j:j_end:stride].transpose(0, 2, 3, 1)
        return cols.reshape(n * out_h * out_w, -1)

    @staticmethod
    def _col2im(cols, x_shape, kernel_size, stride, out_h, out_w):
        n, c, h, w = x_shape
        dX = np.zeros(x_shape, dtype=cols.dtype)
        cols = cols.reshape(n, out_h, out_w, c, kernel_size, kernel_size)
        for i in range(kernel_size):
            i_end = i + stride * out_h
            for j in range(kernel_size):
                j_end = j + stride * out_w
                dX[:, :, i:i_end:stride, j:j_end:stride] += cols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.grads = None
        self.probs = None
        self.labels = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.labels = labels.astype(np.int64)
        batch_size = predicts.shape[0]
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
        eps = 1e-12
        loss = -np.mean(np.log(self.probs[np.arange(batch_size), self.labels] + eps))
        self.grads = self.probs.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # Then send the grads to model for back propagation
        if self.model is not None:
            self.model.backward(self.grads)
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass


class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
