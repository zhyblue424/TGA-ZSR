class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model                
        self.gradients = []           
        self.activations = []                
        self.reshape_transform = reshape_transform 

        self.forward_hook = target_layer.register_forward_hook(self.save_activation) 
        self.backward_hook = target_layer.register_backward_hook(self.save_gradient)  

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu()) 

    def save_gradient(self, module, grad_input, grad_output):

        grad = grad_output[0]

        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu()] + self.gradients 

    def __call__(self, x, x_text):
        self.gradients = []  
        self.activations = []   
        return self.model.forward_mean(x, x_text)  


class ActivationsAndGradients_original:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        target_layer.register_forward_hook(self.save_activation)

        #Backward compitability with older pytorch versions:
        if hasattr(target_layer, 'register_full_backward_hook'):
            target_layer.register_full_backward_hook(self.save_gradient)
        else:
            target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)